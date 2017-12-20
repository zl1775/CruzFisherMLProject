import numpy as np
import re


def load_pfm(file):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = file.readline()
  header = header.decode('utf-8')
  header = header.rstrip()

  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', (file.readline()).decode())
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float((file.readline()).decode().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape), scale


def lookupL(r,c,arr0,arr1):
  disp = arr0[r,c]
  if disp == float('inf'):
    return (0,0)
  else:
    corr = int(c - disp)
    if corr < 0:
      return (1,0)
    elif np.abs(arr1[r,corr] - disp) < 0.5:
      return (2,disp)
    else:
      return (1,0)


def lookupR(r,c,cols,arr0,arr1):
  disp = arr1[r,c]
  if disp == float('inf'):
    return (0,-1)
  else:
    corr = int(c + disp)
    if corr >= cols:
      return (1,-1)
    elif np.abs(arr0[r,corr] - disp) < 0.5:
      return (2,disp)
    else:
      return (1,-1)
