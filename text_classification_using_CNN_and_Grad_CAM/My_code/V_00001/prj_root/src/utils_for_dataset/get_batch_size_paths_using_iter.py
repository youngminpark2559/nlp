import sys,os,copy,argparse
import traceback

def return_bs_paths(iterator,batch_size):
  try:
    bs_paths=[]
    for i in range(batch_size):
      one_path=next(iterator)
      # print("one_path",one_path)
      bs_paths.append(one_path)

    # print("bs_paths",bs_paths)
    # [('/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/trn/0001/CGINTRINSICS_0001_000000_mlt.png',
    #   '/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/ref_srgb_bf2525_dtf1020/0001/CGINTRINSICS_0001_000000_mlt_albedo_srgb_bf2525_dtf1020.png'),
    #  ('/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/trn/0001/CGINTRINSICS_0001_000001_mlt.png',
    #   '/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/ref_srgb_bf2525_dtf1020/0001/CGINTRINSICS_0001_000001_mlt_albedo_srgb_bf2525_dtf1020.png'),
    #  ('/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/trn/0001/CGINTRINSICS_0001_000002_mlt.png',
    #   '/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/ref_srgb_bf2525_dtf1020/0001/CGINTRINSICS_0001_000002_mlt_albedo_srgb_bf2525_dtf1020.png'),
    #  ('/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/trn/0001/CGINTRINSICS_0001_000003_mlt.png',
    #   '/mnt/1T-5e7/papers/cv/IID/CGIntrinsics/code/intrinsics_final2/images/ref_srgb_bf2525_dtf1020/0001/CGINTRINSICS_0001_000003_mlt_albedo_srgb_bf2525_dtf1020.png')]

    return bs_paths

  except:
    print(traceback.format_exc())
    print("Error when loading images")
    print("path_to_be_loaded",path_to_be_loaded)
