def main():
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-south-1")
    s3     = boto3.client("s3", region_name=region)

    print("\n=== HR Analytics — Data Upload to S3 ===")
    print(f"  Bucket  : {S3_BUCKET}")
    print(f"  Project : {PROJECT}")
    print(f"  Region  : {region}")

    create_s3_structure(s3, S3_BUCKET, region)

    # Check if a larger file already exists in S3 — if so, keep it
    resp     = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=RAW_PREFIX)
    s3_files = [o for o in resp.get("Contents", []) if o["Key"].endswith(".csv")]

    if s3_files:
        s3_size  = max(o["Size"] for o in s3_files)
        csv_path = find_csv()
        local_size = os.path.getsize(csv_path)

        if s3_size >= local_size:
            print(f"  S3 already has a file of {s3_size/1024:.1f} KB.")
            print(f"  Local file is {local_size/1024:.1f} KB.")
            print(f"  S3 file is same size or larger — KEEPING S3 file, skipping upload.")
            verify_upload(s3, S3_BUCKET, RAW_PREFIX)
            print("=== Upload skipped — using existing S3 data ===\n")
            return

    # S3 is empty or local file is larger — upload local file
    csv_path = find_csv()
    upload_csv(s3, csv_path, S3_BUCKET, RAW_PREFIX)
    verify_upload(s3, S3_BUCKET, RAW_PREFIX)
    print("=== Upload complete ===\n")