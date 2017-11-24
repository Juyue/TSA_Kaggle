def data_transfer_between_cloud_and_datalab(cloud_file, local_file, download_flag = False, upload_flag = False):
    import os
    os.environ['cloud_file'] = cloud_file
    os.environ['local_file'] = local_file
    if download_file:
        !gcloud cp ${cloud_file} ${local_file}
    if upload_file:
        !gcloud cp ${local_file} ${cloud_file}