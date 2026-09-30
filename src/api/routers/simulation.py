"""Simulation router — inject simulated traffic and attack scenarios for demo/testing."""

import random
import asyncio
from fastapi import APIRouter, Request, BackgroundTasks

router = APIRouter()


def _generate_normal_traffic() -> dict:
    return {
        "duration": random.uniform(0, 10),
        "protocol_type": random.choice(["tcp", "udp"]),
        "service": random.choice(["http", "ftp", "smtp", "ssh"]),
        "flag": "SF",
        "src_bytes": random.randint(100, 5000),
        "dst_bytes": random.randint(0, 3000),
        "land": 0, "wrong_fragment": 0, "urgent": 0,
        "hot": random.randint(0, 2),
        "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
        "num_root": 0, "num_file_creations": 0, "num_shells": 0,
        "num_access_files": 0, "num_outbound_cmds": 0,
        "is_host_login": 0, "is_guest_login": 0,
        "count": random.randint(1, 50),
        "srv_count": random.randint(1, 30),
        "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": random.uniform(0.8, 1.0),
        "diff_srv_rate": random.uniform(0, 0.2),
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 200,
        "dst_host_same_srv_rate": 0.9,
        "dst_host_diff_srv_rate": 0.05,
        "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
        "source_ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
    }


def _generate_dos_attack() -> dict:
    rec = _generate_normal_traffic()
    rec.update({
        "src_bytes": random.randint(500000, 2000000),
        "count": random.randint(400, 512),
        "srv_count": random.randint(400, 512),
        "serror_rate": random.uniform(0.8, 1.0),
        "srv_serror_rate": random.uniform(0.8, 1.0),
        "same_srv_rate": 1.0,
        "dst_host_serror_rate": random.uniform(0.8, 1.0),
        "source_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
    })
    return rec


def _generate_probe_attack() -> dict:
    rec = _generate_normal_traffic()
    rec.update({
        "flag": random.choice(["S0", "REJ"]),
        "logged_in": 0,
        "count": random.randint(200, 512),
        "diff_srv_rate": random.uniform(0.5, 1.0),
        "dst_host_diff_srv_rate": random.uniform(0.3, 1.0),
        "source_ip": f"172.16.{random.randint(0,255)}.{random.randint(1,254)}",
    })
    return rec


@router.post("/simulate/normal", summary="Inject a simulated normal traffic record")
def simulate_normal(request: Request):
    from src.api.routers.detection import detect
    from src.api.models import TrafficRecord
    rec = TrafficRecord(**_generate_normal_traffic())
    return detect(rec, request)


@router.post("/simulate/dos", summary="Inject a simulated DoS attack")
def simulate_dos(request: Request):
    from src.api.routers.detection import detect
    from src.api.models import TrafficRecord
    rec = TrafficRecord(**_generate_dos_attack())
    return detect(rec, request)


@router.post("/simulate/probe", summary="Inject a simulated probe/scan attack")
def simulate_probe(request: Request):
    from src.api.routers.detection import detect
    from src.api.models import TrafficRecord
    rec = TrafficRecord(**_generate_probe_attack())
    return detect(rec, request)


@router.post("/simulate/burst", summary="Inject 50 mixed traffic records for demo")
def simulate_burst(background_tasks: BackgroundTasks, request: Request):
    """Quickly populate the dashboard with varied traffic."""
    generators = [
        (_generate_normal_traffic, 35),
        (_generate_dos_attack, 10),
        (_generate_probe_attack, 5),
    ]

    def _run_burst():
        from src.api.routers.detection import detect
        from src.api.models import TrafficRecord
        for gen, n in generators:
            for _ in range(n):
                try:
                    rec = TrafficRecord(**gen())
                    detect(rec, request)
                except Exception:
                    pass

    background_tasks.add_task(_run_burst)
    return {"status": "burst_started", "total_records": 50}


@router.get("/simulate/stream", summary="Stream simulated attack traffic packet-by-packet via SSE")
async def simulate_stream(
    request: Request,
    scenario: str = "dos",
    count: int = 10,
    delay: float = 0.4
):
    """
    Streams simulated attacks or traffic packet-by-packet in real time.
    Emits Server-Sent Events (SSE) formatted as JSON containing live scores,
    classifications, and automated incident mitigation actions taken.
    """
    from fastapi.responses import StreamingResponse
    import json
    from src.api.routers.detection import detect
    from src.api.models import TrafficRecord

    async def event_generator():
        # Yield initial connection confirmation
        init_event = {
            "type": "init",
            "message": f"Starting real-time simulation: scenario={scenario}, packets={count}",
            "scenario": scenario,
            "count": count,
        }
        yield f"data: {json.dumps(init_event)}\n\n"

        for i in range(count):
            if await request.is_disconnected():
                break

            if scenario == "dos":
                gen = _generate_dos_attack
            elif scenario == "probe":
                gen = _generate_probe_attack
            elif scenario == "normal":
                gen = _generate_normal_traffic
            else:
                gen = random.choice([_generate_normal_traffic, _generate_dos_attack, _generate_probe_attack])

            rec_raw = gen()
            rec = TrafficRecord(**rec_raw)
            result = detect(rec, request)
            res_dict = result.model_dump()
            res_dict["type"] = "packet"
            res_dict["index"] = i + 1
            res_dict["total"] = count
            res_dict["source_ip"] = rec_raw.get("source_ip", "unknown")
            res_dict["protocol"] = rec_raw.get("protocol_type", "tcp")
            res_dict["service"] = rec_raw.get("service", "http")

            yield f"data: {json.dumps(res_dict)}\n\n"
            await asyncio.sleep(delay)

        complete_event = {"type": "complete", "message": "Simulation run complete", "total_sent": count}
        yield f"data: {json.dumps(complete_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

