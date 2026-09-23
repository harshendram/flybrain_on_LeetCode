"""Run some Phase-2 evolution runs on one EC2 instance, with no inbound access and no instance role.

The instance downloads a small payload (the code, circuits, task cache and frozen Phase-1 config; the GitHub repo
is private, so nothing is cloned) through a presigned S3 GET URL, runs `leetfly.experiments.evolve` for the requested
mushroom bodies, and PUTs its results back through presigned URLs (a partial snapshot every 4 minutes, then the final
archive). It then terminates itself. A `shutdown -h +150` timer is a hard backstop against runaway cost.

    python scripts/aws_evolve.py launch --mbs flywire_R flywire_L
    python scripts/aws_evolve.py status
    python scripts/aws_evolve.py fetch        # downloads + unpacks results into results/evolve
    python scripts/aws_evolve.py cleanup      # deletes the bucket objects (instance terminates itself)
"""

import argparse
import io
import json
import tarfile
import time

import boto3
from botocore.config import Config

from leetfly import paths

REGION = "us-east-1"
AMI_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
PINNED = ("numpy==2.4.6 scipy==1.17.1 pandas==3.0.6 pyarrow==25.0.1 scikit-learn==1.9.1 joblib==1.6.0 "
          "pyyaml==6.0.3 threadpoolctl==3.7.0 tqdm")
STATE = paths.RESULTS / "cache" / "aws_job.json"

USER_DATA = """#!/bin/bash
exec > /var/log/leetfly.log 2>&1
set -x
shutdown -h +150
dnf install -y git tar gzip
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
mkdir -p /opt/leetfly && cd /opt/leetfly
curl -fsSL '{payload}' -o payload.tgz && tar xzf payload.tgz && ls
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python {pinned}
uv pip install --python .venv/bin/python -e . --no-deps
nproc; lscpu | grep 'Model name'
( while true; do sleep 240; tar czf /tmp/partial.tgz results/evolve /var/log/leetfly.log; curl -fsS -X PUT -T /tmp/partial.tgz '{partial}'; done ) &
.venv/bin/python -m leetfly.experiments.evolve --jobs {jobs} --mbs {mbs}
tar czf /tmp/results.tgz results/evolve /var/log/leetfly.log
curl -fsS -X PUT -T /tmp/results.tgz '{results}'
shutdown -h now
"""


def s3():
    return boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))


def launch(mbs: list[str], instance_type: str, jobs: int) -> None:
    account = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"leetfly-runs-{account}"
    client = s3()
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=bucket)
        client.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={k: True for k in (
                "BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")},
        )
    job = time.strftime("job-%Y%m%d-%H%M%S")
    buf = io.BytesIO()
    code = [paths.ROOT / "pyproject.toml", paths.ROOT / "configs" / "evolve.yaml",
            *(p for p in (paths.ROOT / "src" / "leetfly").rglob("*.py"))]
    data = [*(paths.PROCESSED / f"mb_{m}_syn5.npz" for m in mbs),
            *paths.PROCESSED.glob("task_v2_*.pkl"),
            paths.RESULTS / "cache" / "phase1_model.pkl"]
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in code + data:
            tar.add(f, arcname=str(f.relative_to(paths.ROOT)).replace("\\", "/"))
    client.put_object(Bucket=bucket, Key=f"{job}/payload.tgz", Body=buf.getvalue())
    ttl = 6 * 3600
    urls = {
        "payload": client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": f"{job}/payload.tgz"}, ExpiresIn=ttl),
        "partial": client.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": f"{job}/partial.tgz"}, ExpiresIn=ttl),
        "results": client.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": f"{job}/results.tgz"}, ExpiresIn=ttl),
    }
    user_data = USER_DATA.format(pinned=PINNED, jobs=jobs, mbs=" ".join(mbs), **urls)
    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    ec2 = boto3.client("ec2", region_name=REGION)
    sg = ec2.create_security_group(GroupName=f"leetfly-{job}", Description="LeetFly batch job: no inbound access")["GroupId"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        InstanceInitiatedShutdownBehavior="terminate",
        SecurityGroupIds=[sg],
        MetadataOptions={"HttpTokens": "required"},
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 16, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"leetfly-evolve-{job}"}]}],
    )
    iid = resp["Instances"][0]["InstanceId"]
    STATE.write_text(json.dumps({"bucket": bucket, "job": job, "instance": iid, "security_group": sg, "mbs": mbs}))
    print(f"launched {iid} ({instance_type}, {ami}) job {job} in bucket {bucket}")


def instance_state(ec2, iid: str) -> str:
    """Terminated instances vanish from DescribeInstances after about an hour."""
    res = ec2.describe_instances(InstanceIds=[iid])["Reservations"]
    return res[0]["Instances"][0]["State"]["Name"] if res else "terminated (no longer listed)"


def status() -> None:
    st = json.loads(STATE.read_text())
    ec2 = boto3.client("ec2", region_name=REGION)
    print(f"instance {st['instance']}: {instance_state(ec2, st['instance'])}")
    client = s3()
    for key in ("partial.tgz", "results.tgz"):
        try:
            head = client.head_object(Bucket=st["bucket"], Key=f"{st['job']}/{key}")
            print(f"  {key}: {head['ContentLength'] / 1e3:.0f} kB, {head['LastModified']:%H:%M:%S}")
        except client.exceptions.ClientError:
            print(f"  {key}: not yet")


def fetch(which: str) -> None:
    st = json.loads(STATE.read_text())
    body = s3().get_object(Bucket=st["bucket"], Key=f"{st['job']}/{which}.tgz")["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        names = tar.getnames()
        runs = [n for n in names if n.startswith("results/evolve/") and n.endswith(".json")]
        for m in tar.getmembers():
            if m.name.startswith("results/evolve/") and m.name.endswith(".json"):
                tar.extract(m, paths.ROOT, filter="data")
            elif m.name.endswith("leetfly.log"):
                (paths.RESULTS / "cache" / f"aws_{which}.log").write_bytes(tar.extractfile(m).read())
    print(f"{which}: {len(runs)} run files unpacked into results/evolve; log in results/cache/aws_{which}.log")


def cleanup() -> None:
    st = json.loads(STATE.read_text())
    client = s3()
    for obj in client.list_objects_v2(Bucket=st["bucket"], Prefix=st["job"] + "/").get("Contents", []):
        client.delete_object(Bucket=st["bucket"], Key=obj["Key"])
    ec2 = boto3.client("ec2", region_name=REGION)
    if not instance_state(ec2, st["instance"]).startswith("terminated"):
        ec2.terminate_instances(InstanceIds=[st["instance"]])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[st["instance"]])
    ec2.delete_security_group(GroupId=st["security_group"])
    print(f"cleaned up job {st['job']}: bucket objects deleted, instance terminated, security group removed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["launch", "status", "fetch", "cleanup"])
    ap.add_argument("--mbs", nargs="+", default=["flywire_R", "flywire_L"])
    ap.add_argument("--instance-type", default="c7a.2xlarge")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--which", default="results", choices=["partial", "results"])
    a = ap.parse_args()
    {"launch": lambda: launch(a.mbs, a.instance_type, a.jobs), "status": status,
     "fetch": lambda: fetch(a.which), "cleanup": cleanup}[a.action]()
