import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as img

IMG_PATH = "cat.jpg"
Y_RANK = 200
CB_RANK = 5
CR_RANK = 5

def main():
  p = img.open(IMG_PATH).convert("YCbCr")
  y, cb, cr = map(lambda x: np.array(x, dtype=np.double), p.split())
  
  p_size = 3 * p.size[0] * p.size[1]
  
  y_svd, cb_svd, cr_svd = map(np.linalg.svd, [y, cb, cr])
  

  channel_approxs = []
  for svd, rank in zip([y_svd, cb_svd, cr_svd], [Y_RANK, CB_RANK, CR_RANK]):
    approx = svd[0][:, :rank] @ np.diag(svd[1][:rank]) @ svd[2][:rank]
    approx[approx > 255] = 255
    approx[approx < 0] = 0
    approx = approx.astype(np.uint8)
    channel_approx = img.fromarray(approx)
    channel_approxs.append(channel_approx)
    
  p_approx = img.merge("YCbCr", channel_approxs)
  p_approx_size = sum([p.size[0] * channel_rank + channel_rank * channel_rank + channel_rank * p.size[1] for channel_rank in [Y_RANK, CB_RANK, CR_RANK]])
  
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

  ax1.imshow(p)
  ax1.set_title("original")
  
  ax2.imshow(p_approx)
  ax2.set_title("compressed\nYCbCr ranks={},\ncompression ratio={:.2%}".format((Y_RANK, CB_RANK, CR_RANK), p_approx_size / p_size))
  
  #fig.savefig("out.svg")
  plt.show()
  
if __name__ == "__main__":
  main()