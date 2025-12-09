import boto3
import uuid
from botocore.exceptions import ClientError

class S3Service:

    def __init__(self, bucket_name: str, region_name: str):
        self.bucket = bucket_name
        self.region = region_name
        self.client = boto3.client("s3", region_name=region_name)

    def upload_bytes(self, content: bytes, key: str, content_type: str):
        try:
            unique_key = f"{uuid.uuid4().hex}-{key}"

            self.client.put_object(
                Bucket=self.bucket,
                Key=unique_key,
                Body=content,
                ContentType=content_type
            )

            return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{unique_key}"

        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}")
