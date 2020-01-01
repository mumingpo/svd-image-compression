import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as img

IMG_PATH = "cat.jpg"
RANKS = [n * 5 for n in range(1, 21)]
NROWS=4
NCOLS=5

def main():
  p = img.open(IMG_PATH).convert("RGB")
  r, g, b = map(lambda x: np.array(x, dtype=np.double), p.split())
  
  p_size = p.size[0] * p.size[1] * 3
  
  r_u, r_s, r_vh = np.linalg.svd(r)
  g_u, g_s, g_vh = np.linalg.svd(g)
  b_u, b_s, b_vh = np.linalg.svd(b)
  
  imgs = []
  
  for rank in RANKS:
    r_approx = r_u[:, :rank] @ np.diag(r_s[:rank]) @ r_vh[:rank]
    r_approx[r_approx > 255] = 255
    r_approx[r_approx < 0] = 0
    r_approx = r_approx.astype(np.uint8)
    
    g_approx = g_u[:, :rank] @ np.diag(g_s[:rank]) @ g_vh[:rank]
    g_approx[g_approx > 255] = 255
    g_approx[g_approx < 0] = 0
    g_approx = g_approx.astype(np.uint8)
    
    b_approx = b_u[:, :rank] @ np.diag(b_s[:rank]) @ b_vh[:rank]
    b_approx[b_approx > 255] = 255
    b_approx[b_approx < 0] = 0
    b_approx = b_approx.astype(np.uint8)

    im = img.merge("RGB", list(map(img.fromarray, [r_approx, g_approx, b_approx])))
    imgs.append(im)
  
  fig, axes = plt.subplots(nrows=NROWS, ncols=NCOLS)
  for row in range(NROWS):
    for col in range(NCOLS):
      idx = row * NCOLS + col
      rank = RANKS[idx]
      
      approx_size = 3 * (p.size[0] * rank + rank * rank + rank * p.size[1])
      
      axes[row][col].imshow(imgs[idx])
      axes[row][col].set_title("rank={},\ncompression ratio={:.2%}".format(rank, approx_size/p_size))
  
  #fig.savefig("out.svg")
  plt.show()
  
if __name__ == "__main__":
  main()