"""Run a Phase-3 (embodied) job on one EC2 instance: flybody physics + TensorFlow 2.8 need Linux and Python 3.10.

Same pattern as scripts/aws_evolve.py: no inbound access, no instance role, code and data travel through presigned
S3 URLs (the GitHub repo is private), partial results every 4 minutes, and the instance terminates itself, with a
hard `shutdown` backstop. The brain code runs from `src/` on PYTHONPATH (it needs no 3.11-only features).

    python scripts/aws_embodied.py launch --module leetfly.embodied.probe --hours 2
    python scripts/aws_embodied.py status
    python scripts/aws_embodied.py fetch [--which partial]   # unpacks results/embodied + the log
    python scripts/aws_embodied.py cleanup
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
FLYBODY = "flybody[tf] @ git+https://github.com/TuragaLab/flybody.git"
EXTRA = "scipy==1.13.1 pandas==2.2.3 pyarrow==17.0.0 scikit-learn==1.5.2 pyyaml joblib tqdm h5py requests"
STATE = paths.RESULTS / "cache" / "aws_embodied_job.json"

USER_DATA = """#!/bin/bash
exec > /var/log/leetfly.log 2>&1
set -x
shutdown -h +{minutes}
dnf install -y git tar gzip mesa-libGL
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
mkdir -p /opt/leetfly && cd /opt/leetfly
curl -fsSL '{payload}' -o payload.tgz && tar xzf payload.tgz && ls
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python "{flybody}" {extra}
.venv/bin/python -c "import tensorflow as tf, flybody, dm_control; print('tf', tf.__version__)"
nproc; lscpu | grep 'Model name'
mkdir -p results/embodied
( while true; do sleep 240; tar czf /tmp/partial.tgz results/embodied /var/log/leetfly.log; curl -fsS -X PUT -T /tmp/partial.tgz '{partial}'; done ) &
export PYTHONPATH=/opt/leetfly/src FLYBODY_DATA=/opt/leetfly/flybody-data TF_CPP_MIN_LOG_LEVEL=2
.venv/bin/python -m {module} {args}
tar czf /tmp/results.tgz results/embodied /var/log/leetfly.log
curl -fsS -X PUT -T /tmp/results.tgz '{results}'
shutdown -h now
"""


def s3():
    return boto3.client("s3", region_name=REGION, config=Config(signature_version="s3v4"))


def launch(module: str, args: str, instance_type: str, hours: float) -> None:
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
    job = time.strftime("emb-%Y%m%d-%H%M%S")
    buf = io.BytesIO()
    files = [paths.ROOT / "pyproject.toml", *paths.ROOT.joinpath("configs").glob("*.yaml"),
             *(p for p in (paths.ROOT / "src" / "leetfly").rglob("*.py")),
             *paths.PROCESSED.glob("mb_*_syn5.npz"), *paths.PROCESSED.glob("task_v2_*.pkl"),
             paths.RESULTS / "cache" / "phase1_model.pkl", paths.RESULTS / "phase1.json"]
    flybody_zips = sorted((paths.RAW / "flybody-data").glob("*.zip"))  # fetched once via a browser (see body.download)
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            if f.exists():
                tar.add(f, arcname=str(f.relative_to(paths.ROOT)).replace("\\", "/"))
        for z in flybody_zips:
            tar.add(z, arcname=f"flybody-data/{z.name}")
    client.put_object(Bucket=bucket, Key=f"{job}/payload.tgz", Body=buf.getvalue())
    ttl = int((hours + 2) * 3600)
    url = lambda op, key: client.generate_presigned_url(op, Params={"Bucket": bucket, "Key": f"{job}/{key}"}, ExpiresIn=ttl)
    user_data = USER_DATA.format(
        minutes=int(hours * 60), payload=url("get_object", "payload.tgz"), partial=url("put_object", "partial.tgz"),
        results=url("put_object", "results.tgz"), flybody=FLYBODY, extra=EXTRA, module=module, args=args,
    )
    ami = boto3.client("ssm", region_name=REGION).get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    ec2 = boto3.client("ec2", region_name=REGION)
    sg = ec2.create_security_group(GroupName=f"leetfly-{job}", Description="LeetFly embodied job: no inbound access")["GroupId"]
    resp = ec2.run_instances(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        InstanceInitiatedShutdownBehavior="terminate",
        SecurityGroupIds=[sg],
        MetadataOptions={"HttpTokens": "required"},
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"leetfly-{job}"}]}],
    )
    iid = resp["Instances"][0]["InstanceId"]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"bucket": bucket, "job": job, "instance": iid, "security_group": sg, "module": module}))
    print(f"launched {iid} ({instance_type}) job {job}: {module} {args} (backstop {hours} h)")


def instance_state(ec2, iid: str) -> str:
    res = ec2.describe_instances(InstanceIds=[iid])["Reservations"]
    return res[0]["Instances"][0]["State"]["Name"] if res else "terminated (no longer listed)"


def status() -> None:
    st = json.loads(STATE.read_text())
    print(f"job {st['job']} ({st['module']}), instance {st['instance']}: {instance_state(boto3.client('ec2', region_name=REGION), st['instance'])}")
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
    n = 0
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.name.startswith("results/embodied/") and m.isfile():
                tar.extract(m, paths.ROOT, filter="data")
                n += 1
            elif m.name.endswith("leetfly.log"):
                (paths.RESULTS / "cache" / f"aws_embodied_{which}.log").write_bytes(tar.extractfile(m).read())
    print(f"{which}: {n} files into results/embodied; log in results/cache/aws_embodied_{which}.log")


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
    print(f"cleaned up {st['job']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["launch", "status", "fetch", "cleanup"])
    ap.add_argument("--module", default="leetfly.embodied.probe")
    ap.add_argument("--args", default="")
    ap.add_argument("--instance-type", default="c7a.2xlarge")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--which", default="results", choices=["partial", "results"])
    a = ap.parse_args()
    {"launch": lambda: launch(a.module, a.args, a.instance_type, a.hours), "status": status,
     "fetch": lambda: fetch(a.which), "cleanup": cleanup}[a.action]()
