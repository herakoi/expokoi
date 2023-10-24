import argparse

import numpy as np
import re

import cv2
import mediapipe as mp

import tkinter
import tkinter.filedialog as filedialog

from pynput import keyboard
from pynput.keyboard import Key

import mido
import rtmidi

import time
import sys

vlims_ = (40,127) 
flims_ = (48, 95) # C2-B5

root = tkinter.Tk()
scrw = root.winfo_screenwidth()
scrh = root.winfo_screenheight()
fill = 1.00
root.withdraw()

global pressed; pressed = None

print(scrw,scrh)

# Convert BGR image into HSV
# -------------------------------------
class gethsv:
  def __init__(self,inp):
    self.bgr = cv2.imread(inp)

    self.h, self.w, _ = self.bgr.shape

    self.hsv = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2HSV)
    
    self.mono = np.logical_and((self.bgr[...,0]==self.bgr[...,1]).all(),
                               (self.bgr[...,1]==self.bgr[...,2]).all())

    if self.mono:
      self.hsv[...,0] = self.hsv[...,2].copy()
    else:
      self.hsv[self.hsv[...,0]>150.00,0] = 0.00

# Convert key name
# Ported from the pretty-midi package
# -------------------------------------
def nametopitch(name):
    pitch_map = {'C': 0, 'D': 2, 'E':  4, 'F':  5, 'G': 7, 'A': 9, 'B': 11}
    accen_map = {'#': 1,  '': 0, 'b': -1, '!': -1}

    try:
      match = re.match(r'^(?P<n>[A-Ga-g])(?P<off>[#b!]?)(?P<oct>[+-]?\d+)$',name)

      pitch = match.group('n').upper()
      offset = accen_map[match.group('off')]
      octave = int(match.group('oct'))
    except:
      raise ValueError('Improper note format: {0}'.format(name))

    return 12*(octave+1)+pitch_map[pitch]+offset

# Keyboard press
# -------------------------------------
def on_press(key):
  global pressed
  if key == Key.esc: pressed = 'esc'
  if key == Key.right: pressed = 'right'
  if key == Key.left:  pressed = 'left'
  if key == Key.up:    pressed = 'up'
  if key == Key.down:  pressed = 'down'

# Build the herakoi player
# =====================================
class start:
  def __init__(self,image=None,mode='single',port={},video=0,box=2,switch=True,**kwargs):
    
    global pressed

    self.listener = keyboard.Listener(on_press=on_press)
    self.listener.start()  

    self.switch = switch

    if image is None:
      tkinter.Tk().withdraw()
      imgpath = image = filedialog.askopenfilenames()
        
      if len(imgpath)<1: sys.exit(1)
    else: imgpath = image

    if isinstance(imgpath,str): imgpath = [imgpath]

    imginit = 0

    modlist = ['single','adaptive']
    if mode not in modlist:
      raise NotImplementedError('"{0}" mode is unknown'.format(mode))
    elif mode=='party':
      raise NotImplementedError('"party mode" not implemented yet')

    modinit = modlist.index(mode)

    self.valname = 'herakoi'

  # Build virtual MIDI port
  # -------------------------------------
    midinew = rtmidi.MidiOut()

    if midinew.get_ports(): midinew.open_port(port.get('value',0))
    else: midinew.open_virtual_port(port.get('name',self.valname))

    self.midiout = mido.open_output(port.get('name',self.valname),virtual=True)

  # Start capture from webcam
  # -------------------------------------
    while True:
      self.opvideo = cv2.VideoCapture(video)
      self.opmusic = gethsv(imgpath[imginit])

      cv2.namedWindow('mixframe',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mixframe',cv2.WND_PROP_FULLSCREEN) 
    # cv2.setWindowProperty('mixframe',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

      self.mphands = mp.solutions.hands
      self.mpdraws = mp.solutions.drawing_utils
      self.mpstyle = mp.solutions.drawing_styles

      self.opindex = 8
      self.opthumb = 4
      
      if mode=='adaptive':
        self.oppatch = None
      else:
        self.oppatch = np.minimum(self.opmusic.w,self.opmusic.h)
        self.oppatch = int(np.clip((box/100)*self.oppatch,2,None))
      
      self.opcolor = {'Left': (0,255,  0), 
                     'Right': (0,255,255)}

      if 'volume' in kwargs:
        vlims = (np.interp(kwargs['volume'],(0,100),(0,127)),127)
      else: vlims = vlims_

      if 'notes' in kwargs:
        flims = (nametopitch(kwargs['notes'][0]),
                 nametopitch(kwargs['notes'][1]))
      else: flims = flims_

      self.run(mode,vlims=vlims,flims=flims,**kwargs)

      if pressed=='esc':
        break
      else:
        if pressed in ['right','left']:
          if  pressed=='right':
            imginit = imginit+1 if imginit<(len(imgpath)-1) else 0
          elif pressed=='left':
            imginit = imginit-1 if imginit>0 else len(imgpath)-1

        if pressed in ['up','down']:
          if pressed=='up':
            modinit = modinit+1 if modinit<(len(modlist)-1) else 0
          elif pressed=='down':
            modinit = modinit-1 if modinit>0 else len(modlist)-1

          mode = modlist[modinit]
          print('changin mode to "{0}"'.format(mode),modinit)

        pressed = None


# Convert H and B to note and loudness
# =====================================
  def getmex(self,posx,box,vlims=vlims_,flims=flims_):
    def getval(img,clip):
      val = np.median(img[np.clip(posx[1]-box[1]//2,0,self.opmusic.h-1):np.clip(posx[1]+box[1]//2,0,self.opmusic.h-1),
                          np.clip(posx[0]-box[0]//2,0,self.opmusic.w-1):np.clip(posx[0]+box[0]//2,0,self.opmusic.w-1)])
      vmidi = 0 if np.isnan(val) else val
      vmidi = int(np.interp(vmidi,(img.min(),img.max()),clip))
      return vmidi

    if self.switch:
      fout = getval(self.opmusic.hsv[...,2],flims)
      vout = getval(self.opmusic.hsv[...,0],vlims)
    else:
      fout = getval(self.opmusic.hsv[...,0],flims)
      vout = getval(self.opmusic.hsv[...,2],vlims)

    return fout, vout

# Draw and return hand markers position
# =====================================
  def posndraw(self,frame,marks,label,draw=True):
    if draw: self.mpdraws.draw_landmarks(frame,marks,self.mphands.HAND_CONNECTIONS,None)

    point = marks.landmark[self.opindex]
    posix = [int(point.x*frame.shape[1]),
             int(point.y*frame.shape[0]),np.abs(point.z)*300]

    if draw and self.oppatch is not None: 
      cv2.circle(frame,(posix[0],posix[1]),np.clip(int(posix[2]),2,None),self.opcolor[label],-1)

    return posix

# Rescale image according to input
# =====================================
  def rescale(self,image):
    if image.shape[1]>image.shape[0]:
      if (self.opmusic.w<self.opmusic.h) or \
         (self.opmusic.h/self.opmusic.w)>(image.shape[0]/image.shape[1]):
        wk = (self.opmusic.w/self.opmusic.h)*image.shape[0]
        wi = int(0.50*(image.shape[1]-wk))     
        return image[:,wi:-wi]
      else: 
        hk = (self.opmusic.h/self.opmusic.w)*image.shape[1]
        hi = int(0.50*(image.shape[0]-hk))
        return image[hi:-hi,:]
    else:
      return image

# Single-user mode
# =====================================
  def run(self,mode='single',vlims=vlims_,flims=flims_,**kwargs):
    global pressed

    imgonly = kwargs.get('imgonly')

    ophands = self.mphands.Hands(max_num_hands=2)

    onmusic = False

    pxshift = kwargs.get('shift',2)
    pxshift = (pxshift/100)*np.minimum(self.opmusic.w,self.opmusic.h)

    toctime = kwargs.get('toc',0.05)
    offtime = kwargs.get('off',0.05)
    tictime = time.time()
    
    newtime = [time.time(),time.time(),2]

    while True:
      _, opframe = self.opvideo.read()

      opframe = self.rescale(opframe)

      opframe = cv2.flip(opframe,1)
      imframe = cv2.cvtColor(opframe,cv2.COLOR_BGR2RGB)

      immusic = self.opmusic.bgr.copy()
      imhands = ophands.process(imframe)

      bhmidif = None
      bhmidiv = None

# ------------------------------------
      
      mixframe = immusic

      fontface  = cv2.FONT_HERSHEY_DUPLEX
      fontscale = 1
      fontcolor = (255,255,255)
      thickness = 1

      text = [['immagine','precedente'],['immagine','successiva']]

      wtmax = 0
      htmax = 0
      for ti in range(len(text)):
        for pi in range(len(text[ti])):
          wtext, htext = cv2.getTextSize(text[ti][pi],fontface,fontscale,thickness)[0]
          wtmax = np.maximum(wtext,wtmax)
          htmax = np.maximum(htext,htmax)

      x1 = [50-int(0.10*wtmax)+ti*int(4*wtmax/3) for ti in range(len(text))]
      y1 = [50-int(0.10*wtmax) for ti in range(len(text))]
      x2 = [50+int(0.10*wtmax)+wtmax+int(4*wtmax/3)*ti for ti in range(len(text))]
      y2 = [55+int(0.10*wtmax)+htmax+int(3*htmax/2) for ti in range(len(text))]

      for ti in range(len(text)):

        button = mixframe[y1[ti]:y2[ti],x1[ti]:x2[ti]].copy()
        weight = np.zeros(button.shape,dtype=button.dtype)

        if time.time()-newtime[ti]>0.10:
          weight[:,:int((time.time()-newtime[ti])*button.shape[1]/newtime[2])] = 255

        button = cv2.addWeighted(button,0.50,weight,0.50,1.00)

        mixframe[y1[ti]:y2[ti],x1[ti]:x2[ti]] = button.copy() 

        for pi in range(len(text[ti])):
          wtext, htext = cv2.getTextSize(text[ti][pi],fontface,fontscale,thickness)[0]
          ptext = (50+int(wtmax/2)-int(wtext/2)+ti*int(4*wtmax/3), 
                   50+int(htmax/2)+int(htext/2)+pi*int(3*htmax/2))

          cv2.putText(mixframe,text[ti][pi],ptext,fontface,fontscale,fontcolor,thickness,cv2.LINE_AA) 

# ------------------------------------

      if imhands.multi_hand_landmarks:
        for mi, immarks in enumerate(imhands.multi_hand_landmarks):
          imlabel = imhands.multi_handedness[mi].classification[0].label

          _       = self.posndraw(opframe,immarks,imlabel,True)

          if self.oppatch is None:
            pxindex = immarks.landmark[self.opindex]
            pxthumb = immarks.landmark[self.opthumb]
            pxpatch = [int(np.abs(pxindex.x-pxthumb.x)*immusic.shape[1]),
                       int(np.abs(pxindex.y-pxthumb.y)*immusic.shape[0])]

            _ = self.posndraw(immusic,immarks,imlabel,True)
            pxmusic = [0.50*(pxindex.x+pxthumb.x),0.50*(pxindex.y+pxthumb.y),0.50*(pxindex.z+pxthumb.z)]

            for im in [immusic,opframe]:
              cv2.rectangle(im,(int(pxthumb.x*im.shape[1]),int(pxthumb.y*im.shape[0])),
                               (int(pxindex.x*im.shape[1]),int(pxindex.y*im.shape[0])),self.opcolor[imlabel],1)
              cv2.circle(im,(int(pxmusic[0]*im.shape[1]),int(pxmusic[1]*im.shape[0])),2,self.opcolor[imlabel],-1)

            pxmusic = [int(pxmusic[0]*immusic.shape[1]),
                       int(pxmusic[1]*immusic.shape[0]),int(pxmusic[2]*300)]

          else:
            pxmusic = self.posndraw(immusic,immarks,imlabel,True)
            pxpatch = [self.oppatch,self.oppatch]

          if (mode in ['single','adaptive'] and imlabel=='Left') or (mode=='party'):
            bhmidif, bhmidiv = self.getmex(pxmusic,pxpatch,vlims,flims)

          if (mode in ['single','adaptive'] and imlabel=='Right'):
            rhmidif, rhmidiv = self.getmex(pxmusic,pxpatch,vlims,flims)

            bhmidif = rhmidif if bhmidif is None else int(0.50*(rhmidif+bhmidif))
            bhmidiv = rhmidiv if bhmidiv is None else int(0.50*(rhmidiv+bhmidiv))

        if (bhmidif is not None) and (bhmidiv is not None):
          if time.time()-tictime>toctime and not onmusic:
            self.midiout.send(mido.Message('note_on',channel=8,note=bhmidif,velocity=bhmidiv))
            pxmusicold = pxmusic
            onmusic = True

          if time.time()-tictime>toctime+offtime and np.hypot(pxmusicold[0]-pxmusic[0],pxmusicold[1]-pxmusic[1])>pxshift:
            self.midiout.send(mido.Message('note_off',channel=8,note=bhmidif))
            self.panic(); onmusic = False
            
            tictime = time.time()

          if x1[0]<=pxmusic[0]<=x2[0] and y1[0]<=pxmusic[1]<=y2[0]:
            if time.time()-newtime[0]>newtime[2]:
              self.panic()
              pressed = 'right'
              break
          else:
            newtime[0] = time.time()

          if x1[1]<=pxmusic[0]<=x2[1] and y1[1]<=pxmusic[1]<=y2[1]:
            if time.time()-newtime[1]>newtime[2]:
              self.panic()
              pressed = 'left'
              break
          else:
            newtime[1] = time.time()
        else: self.panic()
      else: 
        newtime[0] = time.time()
        newtime[1] = time.time()
        self.panic()

      if not imgonly:
        opframe  = cv2.resize(opframe,None,fx=0.20*immusic.shape[0]/opframe.shape[0],fy=0.20*immusic.shape[0]/opframe.shape[0])
        mixframe[                  int(0.01*immusic.shape[0]):opframe.shape[0]+int(0.01*immusic.shape[0]),
                 -opframe.shape[1]-int(0.01*immusic.shape[0]):                -int(0.01*immusic.shape[0])] = opframe

      # Show the combined image in a window
      cv2.imshow('mixframe',mixframe)

      if (cv2.waitKey(1) & 0xFF == ord('q')) or (pressed is not None): break

    self.opvideo.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# Turn off all MIDI notes
# =====================================
  def panic(self):
    for note in range(0,127):
      self.midiout.send(mido.Message('note_off',channel=8,note=note))
    self.midiout.reset()
