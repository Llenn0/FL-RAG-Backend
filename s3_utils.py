import os

# Uses the boto3 AWS API to list the objects in a given bucket
# The prefix argument allows for targeting a specific folder, and the file_type limits returned results to that type
def get_s3_filenames(s3, prefix, file_type):
    response = s3.list_objects_v2(Bucket=os.getenv("AWS_BUCKET_NAME"), Prefix=prefix)
    contents = response.get('Contents', [])
    # List comprehension gets just the filenames and removes incorrect file types
    filenames = [obj['Key'].split('/')[-1] for obj in contents if obj['Key'].endswith(file_type)]
    return filenames