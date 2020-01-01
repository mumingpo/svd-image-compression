import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as img

IMG_PATH = "cat.jpg"
RANKS = [n * 5 for n in range(1, 21)]
NROWS=4
NCOLS=5

def main():
  p = img.open(IMG_PATH).convert("L")
  a = np.array(p, dtype=np.double)
  
  p_size = p.size[0] * p.size[1]
  
  u, s, vh = np.linalg.svd(a)
  
  imgs = []
  
  for rank in RANKS:
    approx = u[:, :rank] @ np.diag(s[:rank]) @ vh[:rank]
    approx[approx > 255] = 255
    approx[approx < 0] = 0
    approx = approx.astype(np.uint8)
    im = img.fromarray(approx)
    imgs.append(im)
  
  fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS)
  for row in range(NROWS):
    for col in range(NCOLS):
      idx = row * NCOLS + col
      rank = RANKS[idx]
      
      approx_size = p.size[0] * rank + rank * rank + rank * p.size[1]
      
      axes[row][col].imshow(imgs[idx])
      axes[row][col].set_title("rank={},\ncompression ratio={:.2%}".format(rank, approx_size/p_size))
  
  #fig.savefig("out.svg")
  plt.show()
  
if __name__ == "__main__":
  main()