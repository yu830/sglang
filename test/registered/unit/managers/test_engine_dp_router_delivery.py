import ipaddress
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import psutil
import zmq
from prometheus_client import CollectorRegistry, Histogram

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers import data_parallel_controller as dp_module
from sglang.srt.managers import delivery_clock, io_struct
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.test.router_delivery_test_utils import RouterFinished as _RouterFinished
from sglang.test.router_delivery_test_utils import create_test_channels
from sglang.test.router_delivery_test_utils import test_port_args as _base_port_args

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_DOMAIN = bytes(range(24))
_OTHER_DOMAIN = bytes(range(24, 48))
_METRIC = "sglang:detokenizer_router_delivery_seconds"


def _test_port_args(endpoint, *, local_endpoint=None):
    ports = _base_port_args(endpoint)
    ports.detokenizer_router_local_endpoint = local_endpoint
    return ports


def _spawn_real_tcp_router(address, worker_endpoints, report, received):
    router = None
    try:
        try:
            domain = delivery_clock.read_clock_domain()
            if domain is None:
                raise AssertionError(
                    "Spawned router could not identify its clock domain"
                )
            registry = CollectorRegistry()

            def registered_histogram(*args, **kwargs):
                return Histogram(*args, registry=registry, **kwargs)

            with patch("prometheus_client.Histogram", side_effect=registered_histogram):
                router = MultiDetokenizerRouter(
                    worker_endpoints,
                    SimpleNamespace(detokenizer_ipc_name="tcp://" + address + ":*"),
                    enable_metrics=True,
                )
            if router.delivery_clock_domain != domain:
                raise AssertionError(
                    "Router did not read the child's actual clock domain"
                )
            router.recv_from_scheduler.setsockopt(zmq.RCVTIMEO, 10000)
            endpoint = router.recv_from_scheduler.getsockopt_string(zmq.LAST_ENDPOINT)
            original_send = router._send
            routed = 0

            def send_then_stop(target, message):
                nonlocal routed
                original_send(target, message)
                routed += 1
                if routed == 3:
                    if not received.wait(10):
                        raise TimeoutError(
                            "Parent did not acknowledge downstream payloads"
                        )
                    raise _RouterFinished

            router._send = send_then_stop
            report.send(
                {
                    "stage": "ready",
                    "domain": domain,
                    "pid": os.getpid(),
                    "endpoint": endpoint,
                    "pickle_ipc": io_struct._USE_PICKLE_IPC,
                }
            )
            try:
                router.event_loop()
            except _RouterFinished:
                pass
            result = {
                "stage": "done",
                "domain": router.delivery_clock_domain,
                "count": registry.get_sample_value(_METRIC + "_count"),
                "sum": registry.get_sample_value(_METRIC + "_sum"),
            }
        finally:
            if router is not None:
                receive_context = router.recv_from_scheduler.context
                mapping_context = router.socket_mapping._zmq_context
                receive_context.destroy(linger=0)
                mapping_context.destroy(linger=0)

        result["resources_closed"] = (
            router.recv_from_scheduler.closed
            and receive_context.closed
            and mapping_context.closed
        )
        report.send(result)
    except BaseException:
        report.send({"stage": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        report.close()


class TestRouterClockDomainCandidate(CustomTestCase):
    def test_native_local_non_loopback_endpoint_is_timestamped(self):
        endpoint = "tcp://192.0.2.1:23123"
        channels = create_test_channels(
            _test_port_args(endpoint, local_endpoint=endpoint)
        )
        self.assertIsNotNone(channels.send_to_detokenizer.clock_domain)
        self.assertEqual(channels.send_to_detokenizer.clock_domain, _DOMAIN)

    def test_non_loopback_endpoint_without_local_candidate_is_not_timestamped(self):
        endpoint = "tcp://192.0.2.1:23123"
        for missing in (False, True):
            with self.subTest(missing=missing):
                ports = _test_port_args(endpoint)
                if missing:
                    del ports.detokenizer_router_local_endpoint
                channels = create_test_channels(ports)
                self.assertIsNone(channels.send_to_detokenizer.clock_domain)

    def test_candidate_does_not_trust_a_changed_endpoint(self):
        ports = _test_port_args(
            "tcp://192.0.2.1:23124", local_endpoint="tcp://192.0.2.1:23123"
        )
        channels = create_test_channels(ports)
        self.assertIsNone(channels.send_to_detokenizer.clock_domain)

    def test_candidate_rejects_non_tcp_and_source_destination_endpoints(self):
        for endpoint in (
            "udp://192.0.2.1:23123",
            "tcp://127.0.0.1:0;192.0.2.1:23123",
        ):
            with self.subTest(endpoint=endpoint):
                channels = create_test_channels(
                    _test_port_args(endpoint, local_endpoint=endpoint)
                )
                self.assertIsNone(channels.send_to_detokenizer.clock_domain)

    def test_unknown_domain_metrics_off_and_nonleader_do_not_time(self):
        endpoint = "tcp://192.0.2.1:23123"
        cases = (
            {"domain": None},
            {"enabled": False},
            {"is_leader": False},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                channels = create_test_channels(
                    _test_port_args(endpoint, local_endpoint=endpoint), **kwargs
                )
                self.assertIsNone(channels.send_to_detokenizer.clock_domain)
                with patch.object(
                    delivery_clock.time,
                    "monotonic_ns",
                    side_effect=AssertionError("disabled sender read the clock"),
                ):
                    channels.send_to_detokenizer.send_output(AbortReq(rid="disabled"))


class _ReachedSchedulerLaunch(Exception):
    pass


class _UnknownEngineBackend(Engine):
    pass


class TestEngineClockDomainCandidate(CustomTestCase):
    def _candidate_at_scheduler_launch(self, **overrides):
        settings = {
            "node_rank": 0,
            "metrics": True,
            "workers": 2,
            "skip": False,
            "joiner": False,
            "rust": False,
            "engine_cls": Engine,
            "placement_group": None,
            "endpoint": "tcp://192.0.2.1:23123",
        }
        settings.update(overrides)
        ports = _test_port_args(settings["endpoint"], local_endpoint="stale-candidate")
        seen = []

        def stop_at_scheduler_launch(_server_args, child_ports, _scheduler, **_kwargs):
            self.assertIs(child_ports, ports)
            seen.append(child_ports.detokenizer_router_local_endpoint)
            raise _ReachedSchedulerLaunch

        getters = {
            "get_parallel": SimpleNamespace(node_rank=settings["node_rank"]),
            "get_exec": SimpleNamespace(
                moe=SimpleNamespace(is_ep_joiner=settings["joiner"])
            ),
            "get_observability": SimpleNamespace(enable_metrics=settings["metrics"]),
            "get_serving": SimpleNamespace(
                detokenizer_worker_num=settings["workers"],
                skip_tokenizer_init=settings["skip"],
                warmups=None,
            ),
            "get_model": SimpleNamespace(
                remote_instance_weight_loader_start_seed_via_transfer_engine=False,
                weight_cache_mode="disabled",
            ),
        }
        startup = (
            "configure_logger",
            "_set_envs_and_config",
            "load_plugins",
            "resolve_auto_parsers",
            "publish",
            "_server_args_for_logging",
        )
        # Stop at scheduler launch while executing Engine's candidate decision.
        with (
            patch.multiple(engine_module, **dict.fromkeys(startup, DEFAULT)),
            patch.multiple(
                engine_module,
                **{
                    name: MagicMock(return_value=value)
                    for name, value in getters.items()
                },
            ),
            patch.object(
                engine_module.envs.SGLANG_RUST_SERVER,
                "get",
                return_value=settings["rust"],
            ),
            patch.object(
                settings["engine_cls"],
                "_launch_scheduler_processes",
                side_effect=stop_at_scheduler_launch,
            ) as launch,
        ):
            with self.assertRaises(_ReachedSchedulerLaunch):
                settings["engine_cls"]._launch_subprocesses(
                    server_args=MagicMock(),
                    init_tokenizer_manager_func=MagicMock(),
                    run_scheduler_process_func=MagicMock(),
                    run_detokenizer_process_func=MagicMock(),
                    port_args=ports,
                    placement_group=settings["placement_group"],
                )
            launch.assert_called_once()
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_native_candidate_is_set_before_scheduler_launch(self):
        self.assertEqual(self._candidate_at_scheduler_launch(), "tcp://192.0.2.1:23123")

    def test_excluded_launches_clear_a_preexisting_candidate(self):
        cases = (
            ("follower", {"node_rank": 1}),
            ("metrics-off", {"metrics": False}),
            ("skip-tokenizer", {"skip": True}),
            ("single-worker", {"workers": 1}),
            ("joiner", {"joiner": True}),
            ("unknown-subclass", {"engine_cls": _UnknownEngineBackend}),
            ("rust", {"rust": True}),
            ("placement-group", {"placement_group": object()}),
            ("ipc", {"endpoint": "ipc:///detokenizer"}),
            ("source-destination", {"endpoint": "tcp://127.0.0.1:0;192.0.2.1:23123"}),
        )
        for name, overrides in cases:
            with self.subTest(name=name):
                self.assertIsNone(self._candidate_at_scheduler_launch(**overrides))


class TestDpClockDomainCandidate(CustomTestCase):
    def test_rebuilt_ports_propagate_only_the_unchanged_router_target(self):
        endpoint = "tcp://192.0.2.1:23123"
        cases = (
            ("same-target", endpoint, False, endpoint, endpoint),
            (
                "different-target",
                "tcp://192.0.2.1:23124",
                False,
                "tcp://192.0.2.1:23124",
                None,
            ),
            (
                "joiner-rewrites-target",
                endpoint,
                True,
                "tcp://192.0.2.2:26002",
                None,
            ),
        )
        for (
            name,
            rebuilt_endpoint,
            joiner,
            expected_endpoint,
            expected_candidate,
        ) in cases:
            with self.subTest(name=name):
                parent_ports = _test_port_args(endpoint, local_endpoint=endpoint)
                parent_ports.nccl_port = 24000
                parent_ports.instance_id = "clock-domain-test"
                rebuilt_ports = _test_port_args(rebuilt_endpoint)
                server_args = MagicMock()
                controller = dp_module.DataParallelController.__new__(
                    dp_module.DataParallelController
                )
                controller.env_lock = threading.Lock()
                controller._scheduler_init_info_lock = threading.Lock()
                controller._scheduler_init_infos = []
                controller.local_kv_event_sources = []
                controller.attn_replica_size = 1
                pipe = MagicMock()
                pipe.recv.return_value = {
                    "max_total_num_tokens": 64,
                    "max_req_input_len": 32,
                }
                controller.prelaunched_schedulers = {(0, 0): (12345, pipe)}
                parallel = SimpleNamespace(
                    attn_dp_enabled=True,
                    pp_size=1,
                    nnodes=1,
                    node_rank=0,
                    tp_size=1,
                    attn_dp_size=1,
                    attn_cp_size=1,
                    moe_dp_size=1,
                    ep_size=1,
                    ep_join_rank_offset=0,
                    dist_init_addr="192.0.2.2:26000",
                )
                execution = SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False),
                    moe=SimpleNamespace(is_ep_scale_joiner=joiner),
                )
                with (
                    patch.object(dp_module, "get_parallel", return_value=parallel),
                    patch.object(dp_module, "get_exec", return_value=execution),
                    patch.object(
                        dp_module,
                        "get_device",
                        return_value=SimpleNamespace(base_gpu_id=0, gpu_id_step=1),
                    ),
                    patch.object(
                        dp_module.PortArgs, "init_new", return_value=rebuilt_ports
                    ) as init_ports,
                    patch.object(dp_module.TorchMemorySaverAdapter, "create"),
                    patch.object(
                        dp_module,
                        "maybe_reindex_device_id",
                        return_value=nullcontext(0),
                    ),
                    patch.object(dp_module.mp, "Process") as process,
                ):
                    controller.launch_tensor_parallel_group(
                        server_args, parent_ports, 0, None, [25006]
                    )
                    process.assert_not_called()
                    init_ports.assert_called_once_with(server_args, 0, [25006])
                pipe.send.assert_called_once()
                scheduler_args, display_ranks = pipe.send.call_args.args[0]
                child_ports = scheduler_args[1]
                self.assertIs(child_ports, rebuilt_ports)
                self.assertEqual(child_ports.detokenizer_ipc_name, expected_endpoint)
                self.assertEqual(
                    child_ports.detokenizer_router_local_endpoint, expected_candidate
                )
                self.assertEqual(child_ports.nccl_port, parent_ports.nccl_port)
                self.assertEqual(child_ports.instance_id, parent_ports.instance_id)

                # Exercise the actual prelaunch receiver rather than inspecting
                # a copied reconstruction of the scheduler argument protocol.
                child_pipe = MagicMock()
                child_pipe.recv.return_value = (scheduler_args, display_ranks)
                scheduler_entry = MagicMock()
                with patch.object(dp_module, "kill_itself_when_parent_died"):
                    dp_module.run_scheduler_process_with_init_pipe(
                        scheduler_entry, child_pipe
                    )
                scheduler_entry.assert_called_once_with(
                    *scheduler_args, child_pipe, *display_ranks
                )
                self.assertIs(scheduler_entry.call_args.args[1], child_ports)
                child_pipe.close.assert_called_once()

    def test_pure_dp_rebuild_passes_router_endpoint_and_candidate_to_thread(self):
        endpoint = "tcp://192.0.2.1:23123"
        for candidate in (endpoint, None):
            with self.subTest(candidate=candidate):
                parent_ports = _test_port_args(endpoint, local_endpoint=candidate)
                parent_ports.instance_id = "pure-dp-clock-domain-test"
                rebuilt_ports = _test_port_args(
                    "tcp://192.0.2.1:23124",
                    local_endpoint="stale-rebuilt-candidate",
                )
                rebuilt_ports.nccl_port = 24000
                server_args = MagicMock()
                controller = dp_module.DataParallelController.__new__(
                    dp_module.DataParallelController
                )
                controller.context = MagicMock()
                controller.workers = [None]
                parallel = SimpleNamespace(
                    num_dp_ranks=1, tp_size=1, pp_size=1, node_rank=0
                )
                with (
                    patch.object(dp_module, "get_parallel", return_value=parallel),
                    patch.object(
                        dp_module,
                        "get_device",
                        return_value=SimpleNamespace(gpu_id_step=1),
                    ),
                    patch.object(
                        dp_module.PortArgs, "init_new", return_value=rebuilt_ports
                    ) as init_ports,
                    patch.object(dp_module, "bind_port") as bind_port,
                    patch.object(dp_module, "get_zmq_socket") as get_socket,
                    patch.object(dp_module.threading, "Event") as event,
                    patch.object(dp_module.threading, "Thread") as thread,
                ):
                    controller.launch_dp_schedulers(server_args, parent_ports)
                    init_ports.assert_called_once_with(server_args)
                    bind_port.assert_called_once_with(rebuilt_ports.nccl_port)
                    bind_port.return_value.close.assert_called_once()
                    thread.assert_called_once()
                    self.assertEqual(
                        thread.call_args.kwargs["target"],
                        controller.launch_tensor_parallel_group_thread,
                    )
                    child_args = thread.call_args.kwargs["args"]
                    self.assertEqual(
                        child_args,
                        (server_args, rebuilt_ports, 0, 0, event.return_value),
                    )
                    child_ports = child_args[1]
                    self.assertEqual(child_ports.detokenizer_ipc_name, endpoint)
                    self.assertEqual(
                        child_ports.detokenizer_router_local_endpoint, candidate
                    )
                    self.assertEqual(child_ports.instance_id, parent_ports.instance_id)
                    get_socket.assert_called_once_with(
                        controller.context,
                        zmq.PUSH,
                        rebuilt_ports.scheduler_input_ipc_name,
                        True,
                    )
                    thread.return_value.start.assert_called_once()
                    event.return_value.wait.assert_called_once()


class TestSpawnedRouterClockDomain(CustomTestCase):
    def _local_ipv4_address(self):
        if sys.platform != "linux":
            self.skipTest("Real boot/time-namespace identities require Linux")
        addresses = [
            item.address
            for entries in psutil.net_if_addrs().values()
            for item in entries
            if item.family == socket.AF_INET
            and not ipaddress.ip_address(item.address).is_loopback
            and not ipaddress.ip_address(item.address).is_unspecified
        ]
        if not addresses:
            self.skipTest("No local non-loopback IPv4 interface is available")
        return addresses[0]

    @staticmethod
    def _read_report(report, process, timeout):
        if not report.poll(timeout):
            raise TimeoutError(
                f"Router report timed out: pid={process.pid}, "
                f"exitcode={process.exitcode}"
            )
        result = report.recv()
        if result["stage"] == "error":
            raise AssertionError(result["traceback"])
        return result

    def _check_ready(self, ready, process, address, domain):
        self.assertEqual(ready["stage"], "ready")
        self.assertEqual(ready["pid"], process.pid)
        self.assertNotEqual(ready["pid"], os.getpid())
        self.assertEqual(ready["domain"], domain)
        self.assertEqual(ready["pickle_ipc"], io_struct._USE_PICKLE_IPC)
        self.assertTrue(ready["endpoint"].startswith("tcp://" + address + ":"))

    def _check_forwarded_messages(self, sender, workers):
        sender.socket.setsockopt(zmq.SNDTIMEO, 5000)
        messages = [
            AbortReq(rid="same-domain", abort_message="matching clock"),
            AbortReq(rid="different-domain", abort_message="different clock"),
            AbortReq(rid="unknown-domain", abort_message="no clock identity"),
        ]
        sender.send_output(messages[0])
        delivery_clock.sock_send_with_timestamp(
            sender.socket, messages[1], clock_domain=_OTHER_DOMAIN
        )
        io_struct.sock_send(sender.socket, messages[2])

        poller = zmq.Poller()
        for worker in workers:
            poller.register(worker, zmq.POLLIN)
        forwarded = {}
        target_by_rid = {}
        for _ in messages:
            readable = dict(poller.poll(15000))
            self.assertTrue(readable, "Missing downstream payload")
            worker = next(iter(readable))
            message = io_struct.sock_recv(worker)
            self.assertNotIn(message.rid, forwarded)
            forwarded[message.rid] = message
            target_by_rid[message.rid] = workers.index(worker)

        self.assertEqual(forwarded, {message.rid: message for message in messages})
        self.assertEqual(
            target_by_rid,
            {"same-domain": 0, "different-domain": 1, "unknown-domain": 1},
        )

    def _check_done(self, done, domain):
        self.assertEqual(done["stage"], "done")
        self.assertEqual(done["domain"], domain)
        self.assertEqual(done["count"], 1.0)
        self.assertGreaterEqual(done["sum"], 0.0)
        self.assertLess(done["sum"], 30.0)
        self.assertTrue(done["resources_closed"])

    @staticmethod
    def _cleanup_router(process, report, child_report, received, workers, channels):
        received.set()
        sender_context = None
        if channels is not None:
            sender_context = channels.send_to_detokenizer.socket.context
            sender_context.destroy(linger=0)
        for worker in workers:
            worker.close(0)

        terminated = False
        if process.pid is not None:
            process.join(timeout=10)
            if process.is_alive():
                terminated = True
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

        child_errors = []
        while report.poll(0):
            try:
                pending = report.recv()
            except EOFError:
                break
            if pending["stage"] == "error":
                child_errors.append(pending["traceback"])
        child_report.close()
        report.close()
        child_pid = process.pid
        child_exitcode = process.exitcode
        child_alive = process.is_alive()
        if child_pid is not None and not child_alive:
            process.close()
        return SimpleNamespace(
            pid=child_pid,
            exitcode=child_exitcode,
            alive=child_alive,
            terminated=terminated,
            errors=child_errors,
            sender_context=sender_context,
        )

    def test_spawned_router_uses_real_domain_and_non_loopback_tcp(self):
        address = self._local_ipv4_address()
        parent_domain = delivery_clock.read_clock_domain()
        self.assertIsInstance(parent_domain, bytes)
        self.assertEqual(len(parent_domain), 24)

        process_context = mp.get_context("spawn")
        report, child_report = process_context.Pipe(duplex=False)
        received = process_context.Event()
        channels = None
        failure = None
        with (
            tempfile.TemporaryDirectory(prefix="sglang-spawn-clock-") as tmp,
            zmq.Context() as worker_context,
        ):
            workers = [
                worker_context.socket(zmq.PULL),
                worker_context.socket(zmq.PULL),
            ]
            worker_endpoints = [
                "ipc://" + tmp + "/worker-a",
                "ipc://" + tmp + "/worker-b",
            ]
            for worker, endpoint in zip(workers, worker_endpoints):
                worker.setsockopt(zmq.LINGER, 0)
                worker.bind(endpoint)
            process = process_context.Process(
                target=_spawn_real_tcp_router,
                args=(address, worker_endpoints, child_report, received),
            )
            try:
                process.start()
                child_report.close()
                ready = self._read_report(report, process, 45)
                self._check_ready(ready, process, address, parent_domain)
                endpoint = ready["endpoint"]
                ports = _test_port_args(endpoint, local_endpoint=endpoint)
                for field in (
                    "scheduler_input_ipc_name",
                    "rpc_ipc_name",
                    "tokenizer_ipc_name",
                    "metrics_ipc_name",
                ):
                    setattr(ports, field, "ipc://" + tmp + "/" + field)
                channels = SchedulerIpcChannels.create(
                    port_args=ports,
                    is_rank_zero=True,
                    skip_tokenizer_init=False,
                    metrics_enabled=True,
                    enable_scripted_runtime=False,
                    detokenizer_router_metrics_enabled=True,
                )
                self.assertIsNotNone(channels.send_to_detokenizer.clock_domain)
                self.assertEqual(
                    channels.send_to_detokenizer.clock_domain, parent_domain
                )
                self._check_forwarded_messages(channels.send_to_detokenizer, workers)
                received.set()
                done = self._read_report(report, process, 15)
                self._check_done(done, parent_domain)
            except Exception as exc:
                failure = exc
            finally:
                cleanup = self._cleanup_router(
                    process, report, child_report, received, workers, channels
                )

            diagnostic = (
                f"router pid={cleanup.pid}, exitcode={cleanup.exitcode}, "
                f"terminated_after_timeout={cleanup.terminated}"
            )
            if cleanup.errors:
                diagnostic += "\n" + "\n".join(cleanup.errors)
            if failure is not None:
                raise AssertionError(f"{failure}\n{diagnostic}") from failure
            self.assertFalse(cleanup.terminated, diagnostic)
            self.assertFalse(cleanup.alive, diagnostic)
            self.assertEqual(cleanup.exitcode, 0, diagnostic)
            self.assertFalse(cleanup.errors, diagnostic)
            self.assertTrue(all(worker.closed for worker in workers))
            self.assertTrue(cleanup.sender_context.closed)
            self.assertTrue(report.closed)
            self.assertTrue(child_report.closed)
        self.assertTrue(worker_context.closed)


if __name__ == "__main__":
    unittest.main()
