#! /usr/local/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import cgi
import cv2
from units import *
import pickle

form = cgi.FieldStorage()
Hol = 8
f = open("data/img_path.txt","rb")
imglist = pickle.load(f)
file_n = 0
feature = 0
calculation = 0
num = 0

feature_name = ["RGB Histogram 1x1", "RGB Histogram 2x2", "RGB Histogram 3x3", "HSV Histogram 1x1", "HSV Histogram 2x2", "HSV Histogram 3x3", "LUV Histogram 1x1", "LUV Histogram 2x2", "LUV Histogram 3x3", "VGG16 fc7 feature"]
calculation_name = ["Intersection", "Euclid"]

if ( "file_n" in form ):
    try:
        file_n = int(form["file_n"].value)
    except:
        file_n = 0
if ( "feature" in form):
    try:
        feature = int(form["feature"].value)
    except:
        feature = 0

if ( "calculation" in form ):
    try:
        calculation = int(form["calculation"].value)
    except:
        calculation = 0

print("Content-Type: text/html\n")

if feature ==1:
  feature_data = np.load('data/rgb_hists2x2.npy')
elif feature ==2:
  feature_data = np.load('data/rgb_hists3x3.npy')
elif feature ==3:
  feature_data = np.load('data/hsv_hists1x1.npy')
elif feature ==4:
  feature_data = np.load('data/hsv_hists2x2.npy')
elif feature ==5:
  feature_data = np.load('data/hsv_hists3x3.npy')
elif feature ==6:
  feature_data = np.load('data/luv_hists1x1.npy')
elif feature ==7:
  feature_data = np.load('data/luv_hists2x2.npy')
elif feature ==8:
  feature_data = np.load('data/luv_hists3x3.npy')
elif feature ==9:
  feature_data = np.load('data/dcnnf.npy')
else:
  feature_data = np.load('data/rgb_hists1x1.npy')

if calculation==1:
  dist_data = Euclid(feature_data[file_n], feature_data)
  sort_idx = Sort(dist_data, 0)
else:
  dist_data = Intersection(feature_data[file_n], feature_data)
  sort_idx = Sort(dist_data, 1)

html1 = """
<html>
  <head>
    <meta charset="UTF-8">
    <title>柳井研究室課題3a</title>
    <style>
      .trim {
        overflow: hidden;
        width: 200px;
        height: 200px;
        position: relative;
      }
      .trim img {
        position: absolute;
        top: 50%;
        left: 50%;
        -webkit-transform: translate(-50%, -50%);
        -ms-transform: translate(-50%, -50%);
        transform: translate(-50%, -50%);
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
      }
  </style>
  </head>
  <body>
"""
print(html1)

html2 ="""
    <div style="text-align:center; font-size:xx-large">
      Search by %s with %s<br>
    </div>
"""
print(html2 % (feature_name[feature], calculation_name[calculation]))

html3 = """
    <div style="text-align:center; font-size:large">
      [Intersection]<br>
      <a href = "./?file_n=%s&feature=0&calculation=0"> RGB Histogram 1x1</a>
      <a href = "./?file_n=%s&feature=1&calculation=0"> RGB Histogram 2x2</a>
      <a href = "./?file_n=%s&feature=2&calculation=0"> RGB Histogram 3x3</a><br>
      <a href = "./?file_n=%s&feature=3&calculation=0"> HSV Histogram 1x1</a>
      <a href = "./?file_n=%s&feature=4&calculation=0"> HSV Histogram 2x2</a>
      <a href = "./?file_n=%s&feature=5&calculation=0"> HSV Histogram 3x3</a><br>
      <a href = "./?file_n=%s&feature=6&calculation=0"> LUV Histogram 1x1</a>
      <a href = "./?file_n=%s&feature=7&calculation=0"> LUV Histogram 2x2</a>
      <a href = "./?file_n=%s&feature=8&calculation=0"> LUV Histogram 3x3</a><br>
      <a href = "./?file_n=%s&feature=9&calculation=0"> VGG16 fc7 feature</a><br>
      [Euclid]<br>
      <a href = "./?file_n=%s&feature=0&calculation=1"> RGB Histogram 1x1</a>
      <a href = "./?file_n=%s&feature=1&calculation=1"> RGB Histogram 2x2</a>
      <a href = "./?file_n=%s&feature=2&calculation=1"> RGB Histogram 3x3</a><br>
      <a href = "./?file_n=%s&feature=3&calculation=1"> HSV Histogram 1x1</a>
      <a href = "./?file_n=%s&feature=4&calculation=1"> HSV Histogram 2x2</a>
      <a href = "./?file_n=%s&feature=5&calculation=1"> HSV Histogram 3x3</a><br>
      <a href = "./?file_n=%s&feature=6&calculation=1"> LUV Histogram 1x1</a>
      <a href = "./?file_n=%s&feature=7&calculation=1"> LUV Histogram 2x2</a>
      <a href = "./?file_n=%s&feature=8&calculation=1"> LUV Histogram 3x3</a><br>
      <a href = "./?file_n=%s&feature=9&calculation=1"> VGG16 fc7 feature</a><br>
    </div>
    <div style="text-align:center; font-size:large">
      <a href = "./?file_n=%s&feature=%s&calculation=%s"><img src="%s" width="500px">
    </div>
    <table align="center" border="1" cellpadding="0" cellspacing="4">
"""


print(html3 % (str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(file_n), str(feature), str(calculation), imglist[file_n]))
for i in range(len(imglist)):
  if i%Hol == 0:
    print('<tr>')
  html4 = """
    <td width="12.5%%" align="center"><a href = "./?file_n=%s&feature=%s&calculation=%s"><div class="trim"><img src="%s" border="0"></div></a><br>
    [%s] %f</td>
  """
  img = cv2.imread(imglist[sort_idx[i]])
  print(html4 % (str(sort_idx[i]), str(feature), str(calculation), imglist[sort_idx[i]], str(sort_idx[i]), dist_data[sort_idx[i]]))
  if i%Hol == Hol-1:
    print("</tr>")

print("</table></body></html>")