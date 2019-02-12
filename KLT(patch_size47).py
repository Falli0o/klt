
# coding: utf-8

# In[19]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[20]:


cap = cv2.VideoCapture('AasaDukkehus.mp4')
wid = int(cap.get(3))
hei = int(cap.get(4))
framerate = int(cap.get(5))
framenum = int(cap.get(7))
 
video = np.empty((framenum,hei,wid,3),dtype=np.uint8)#dtype='float64')
cnt = 0
while cnt<framenum:
    a,b=cap.read()
    #cv2.imshow('%d'%cnt, b)
    #cv2.waitKey(20)
    #b = b#.astype('float64')/255
    video[cnt]=b#.astype(np.uint8)
    #print(cnt)
    cnt+=1
print ('done')
#references:https://blog.csdn.net/yuejisuo1948/article/details/80734908 


# In[21]:


def harris_corner(gray_img,patch_size,sigma,min_dst,min_n = 500,k=0.05):#harris corner
    h = gray_img.shape[0]
    w = gray_img.shape[1]
    gy,gx = np.gradient(gray_img)

    gx_square = cv2.GaussianBlur(gx**2,(patch_size,patch_size),sigma)
    gx_multiply_gy = cv2.GaussianBlur(np.multiply(gx,gy),(patch_size,patch_size),sigma)
    gy_square = cv2.GaussianBlur(gy**2,(patch_size,patch_size),sigma)
    detA = gx_square * gy_square - gx_multiply_gy ** 2
    # trace
    traceA = gx_square + gy_square
    #k = 0.05
    R = detA - k * (traceA ** 2)
    ind = np.unravel_index(np.argsort(R, axis=None)[::-1], R.shape)
    row,col = ind
    points = np.vstack([row,col]).T
    good_points = []
    rs = []
    for p in range(points.shape[0]):
        point = points[p]
        one_row,one_col = point
        r = R[one_row,one_col]
        if len(good_points) == 0:
            if ((min_dst-1) < one_row < (h-1-min_dst)) and ((min_dst-1) < one_col < (w-1-min_dst)):
                good_points.append(point)
        else:
            gg_points = np.array(good_points)
            times = gg_points.shape[0]
            m = np.vstack((np.repeat(point[0],times),np.repeat(point[1],times))).T
            check_x = (np.abs(gg_points[:,0] - m[:,0])> min_dst).all()
            check_y = (np.abs(gg_points[:,1] - m[:,1])> min_dst).all()
            if np.array([check_x,check_y]).any() == True:
                if ((min_dst-1) < one_row < (h-1-min_dst)) and ((min_dst-1) < one_col < (w-1-min_dst)):
                    good_points.append(point)
                    if len(good_points)%100 == 0:
                        print (len(good_points))
                    rs.append(r)
        if len(good_points) >= min_n: #print 10
            #print (len(good_points))
            break
    return (np.array(good_points),np.array(rs))


# In[22]:


g = cv2.GaussianBlur(cv2.cvtColor(video[0],cv2.COLOR_BGR2GRAY).astype('float32')/255,(7,7),1)


# In[23]:


corners,Rs = harris_corner(g,patch_size=17,sigma=3,min_dst=3,min_n = 500,k=0.06)
corners = np.vstack((corners[:,1],corners[:,0])).T
Rs.min()
#4 450


# In[24]:


fig = plt.figure()
fig.set_size_inches(15, 7);
plt.imshow(video[0][...,::-1])
plt.scatter(corners[:,0],corners[:,1],s = 4)


# In[25]:


def gaussian_kernel(shape, sigma ):
    x, y = np.meshgrid(np.linspace(-1,1,shape[0]), np.linspace(-1,1,shape[1]))
    d = np.sqrt(x*x+y*y)
    mu = 0.0
    g = np.exp(-((d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g
gaussian2 = gaussian_kernel((47,47),8)
plt.imshow(gaussian2)


# In[26]:


def cal_d(g1,g2,corners_set,width,sigma,w='gaussian'):
    #p2s = []
    res = []
    new_c = []
    dets = []
    last_d = []
    length = len(corners_set)
    for p in range(length):
        #print (p)
        movex = 0
        movey = 0
        dy = 0
        dx = 0
        i = 0
        x,y = corners_set[p]
        p1 = cv2.getRectSubPix(g1, (width,width), (x,y))
        gy,gx = np.gradient(p1)
        gx2 = gx*gx
        gy2 = gy*gy
        gxy = gx*gy
        gau = gaussian_kernel((width,width), sigma)
        if w == 'gaussian':
            ggx2 = np.multiply(gx2,gau)
            ggy2 = np.multiply(gy2,gau)
            ggxy = np.multiply(gxy,gau)
            G = np.array([[ggx2.sum(),ggxy.sum()],[ggxy.sum(),ggy2.sum()]])
        else:
            G = np.array([[gx2.sum(),gxy.sum()],[gxy.sum(),gy2.sum()]])
        det = np.linalg.det(G)
        if det <= 0:
            new_c.append([0,0])
            res.append(100)
            last_d.append(100)
        else:
            
            G_inv = np.linalg.inv(G)


            #p2s = []
            while i < 20:
                p2 = cv2.getRectSubPix(g2, (width,width), (x,y))
                re = np.sum((p2 - p1)**2)
                #p2s.append(p2)
                diff = p1 - p2
                diffx = diff*gx
                diffy = diff*gy
                if w == 'gaussian':
                    gdiffx = np.multiply(diffx,gau)
                    gdiffy = np.multiply(diffy,gau)
                    E = np.array([[gdiffx.sum()],[gdiffy.sum()]])
                else:
                    E = np.array([[diffx.sum()],[diffy.sum()]])
                d = np.dot(G_inv,E).reshape(-1)
                dx,dy = d
                y = y+dy
                x = x+dx
                movex += dy
                movey += dx
                #print (d)
                i += 1
            move = np.array([movex,movey]).max()
            #print (move)
            re = np.sum((p2 - p1)**2)
            if re < 100.0:
                last_d.append(np.abs(d).max())
                res.append(re)
                new_c.append([x,y])
            else:
                new_c.append([0,0])
                res.append(100)
                last_d.append(100)
            
        p += 1
        #print ('===============')
    return (np.array(new_c),np.array(res),np.array(last_d),np.array(dets))


# In[27]:


f_corners = corners
frame = 0
all_fc = [corners]
all_fp = [corners]
rr = []
dd = []
while frame < (framenum-1):
    g1 = cv2.GaussianBlur(cv2.cvtColor(video[frame],cv2.COLOR_BGR2GRAY).astype('float32')/255,(7,7),1)   
    g2 = cv2.GaussianBlur(cv2.cvtColor(video[frame+1],cv2.COLOR_BGR2GRAY).astype('float32')/255,(7,7),1)

    
    #print ('f: %d' % (frame+1))
    cs,r,d,_ = cal_d(g1,g2,f_corners,47,sigma=8,w='gaussian')
    all_fc.append(cs)
    rr.append(r)
    dd.append(d)
    f_corners = cs
    frame += 1


# In[28]:



def cal_G(grayimg,c,width,sigma,w):
    x,y = c
    p = cv2.getRectSubPix(grayimg, (width,width), (x,y))
    gy,gx = np.gradient(p)
    gx2 = gx*gx
    gy2 = gy*gy
    gxy = gx*gy
    gau = gaussian_kernel((width,width), sigma)
    if w == 'gaussian':
        ggx2 = np.multiply(gx2,gau)
        ggy2 = np.multiply(gy2,gau)
        ggxy = np.multiply(gxy,gau)
        G = np.array([[ggx2.sum(),ggxy.sum()],[ggxy.sum(),ggy2.sum()]])
    else:
        G = np.array([[gx2.sum(),gxy.sum()],[gxy.sum(),gy2.sum()]])
    return G


# In[29]:


ava_fc = []
for f in range(84):
    #print (f)
    cs = all_fc[f]
    min_evs = []
    new_c = []
    for p in cs:
        x,y = p
        G = cal_G(g,p,47,8,w='gaussian')
        det = np.linalg.det(G)
        trace = np.trace(G)
        r = det - 0.06*(trace**2)
        ev = np.linalg.eig(G)[0].min()
        if (r<Rs.min()*0.8):#(ev > 0.001) and 
            new_c.append(np.array([x,y]))
        else:
            new_c.append(np.array([0,0]))
    ava_fc.append(np.array(new_c))


# In[30]:


#num = []
#for i in range(83):
 #   x = all_fc[i][:,0]
  #  y = all_fc[i][:,1]
   # idx = (x > 0) & (x<1920) & (y > 0) & (y<1080)
    #n = np.unique(all_fc[i][idx],axis=0).shape[0]
    #num.append(n)


# In[31]:


all_fc = np.array(all_fc)
np.save('p47.npy',all_fc)


# In[32]:


#fig = plt.figure()
#fig.set_size_inches(15, 7);
#plt.scatter(range(1,84),num,marker='x',color = 'green')
#plt.plot(range(1,84),num,color = 'green')
#plt.grid(linestyle='--')
#plt.yticks([j for j in range(0,550,50)] + [21]);
#plt.xticks([1] + [j for j in range(10,90,10)] + [83]);


# In[33]:


for f in range(83):
    fig = plt.figure()
    fig.set_size_inches(15, 7);
    fig.set_visible(True)
    #g = cv2.GaussianBlur(cv2.cvtColor(video[f],cv2.COLOR_BGR2GRAY).astype('float32')/255,(15,15),3)

    
    plt.imshow(video[f][...,::-1],cmap='gray');
    #plt.scatter(corners[:,1],corners[:,0],s =4)
    #plt.scatter(new_c11[:,1],new_c11[:,0],s =4)
    #sa = (ava_fc[f+1] - ava_fc[f])**2
    sa = (all_fc[f+1] - all_fc[f])**2
    dis = np.sqrt(np.sum(sa,axis=-1))
    idx = np.where(dis < 40)
    #all_fc[f][idx]
    #plt.scatter(ava_fc[f][idx][:,0],ava_fc[f][idx][:,1],color='yellow',s=7);
    plt.scatter(all_fc[f][:,0],all_fc[f][:,1],color='yellow',s=7);
    #plt.scatter(ava_fc[f][:,0],ava_fc[f][:,1],color='yellow',s=7);
    #plt.scatter(all_fc[f][idx][:,0],all_fc[f][idx][:,1],color='yellow',s=7);
    #plt.scatter(merge_c[:,1],merge_c[:,0],s=4)
    #plt.quiver(all_fc[f][:,0],all_fc[f][:,1], 
               #all_fc[f+1][:,0]-all_fc[f][:,0], all_fc[f+1][:,1]-all_fc[f][:,1],color='green',scale=35,scale_units='inches');
    plt.quiver(all_fc[f][idx][:,0],all_fc[f][idx][:,1], all_fc[f+1][idx][:,0]-all_fc[f][idx][:,0],
               all_fc[f+1][idx][:,1]-all_fc[f][idx][:,1],color='green',scale=35,scale_units='inches');
    #plt.quiver(ava_fc[f][idx][:,0],ava_fc[f][idx][:,1], 
               #ava_fc[f+1][idx][:,0]-ava_fc[f][idx][:,0], ava_fc[f+1][idx][:,1]-ava_fc[f][idx][:,1],color='green');
    plt.xlim([0,1920]);
    plt.ylim([1080,0]);
    plt.axis('off');
    plt.savefig(r'.\vec47\%d.png' % (f+1))


# In[34]:


for f in range(83):
    fig = plt.figure()
    fig.set_size_inches(15, 7);
    fig.set_visible(True)
    #g = cv2.GaussianBlur(cv2.cvtColor(video[f],cv2.COLOR_BGR2GRAY).astype('float32')/255,(15,15),3)    
    plt.imshow(video[f][...,::-1],cmap='gray');
    #plt.scatter(corners[:,1],corners[:,0],s =4)
    #plt.scatter(new_c11[:,1],new_c11[:,0],s =4)
    plt.scatter(all_fc[f][:,0],all_fc[f][:,1],color= 'limegreen',marker = 'x',s=37);
    #plt.scatter(merge_c[:,1],merge_c[:,0],s=4)
    #plt.quiver(all_fc[f][:,1],all_fc[f][:,0], all_fc[f+1][:,1]-all_fc[f][:,1], all_fc[f+1][:,0]-all_fc[f][:,0],color='green');
    plt.xlim([0,1920]);
    plt.ylim([1080,0]);
    plt.axis('off');
    plt.savefig(r'.\p47\%d.png' % (f+1))


# In[35]:


fourcc = cv2.VideoWriter_fourcc(*'DIVX')#'M','J','P','G')#cv2.destroyAllWindows()
vout = cv2.VideoWriter('motion_vector47.avi',fourcc,15,(1080, 504),True)
#vout.open()

for i in range(83):
    img = cv2.imread(r'C:\Users\wzy\Desktop\AIA\1\vec47\%d.png' % (i+1))
    #cv2.imshow('frame',img)
    vout.write(img)
cv2.destroyAllWindows()
vout.release()


# In[36]:


fourcc = cv2.VideoWriter_fourcc(*'DIVX')#'M','J','P','G')#cv2.destroyAllWindows()
vout = cv2.VideoWriter('p47.avi',fourcc,15,(1080, 504),True)
#vout.open()

for i in range(83):
    img = cv2.imread(r'C:\Users\wzy\Desktop\AIA\1\p47\%d.png' % (i+1))
    #cv2.imshow('frame',img)
    vout.write(img)
cv2.destroyAllWindows()
vout.release()

