import argparse

import numpy as np
import re

import cv2
import mediapipe as mp
import pygame

import tkinter
import tkinter.filedialog as filedialog

from pynput import keyboard
from pynput.keyboard import Key

from mingus.midi import fluidsynth
from mingus.midi import pyfluidsynth

import time
import sys
import os

vlims_ = (40,127) 
flims_ = (48, 95) # C2-B5

root = tkinter.Tk()
scrw = root.winfo_screenwidth()
scrh = root.winfo_screenheight()
fill = 1.00
root.withdraw()

pygame.init()

global pressed; pressed = None
global  screen; screen  = None

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
  if key == Key.esc:   pressed = 'esc'

# Build the expokoi player
# =====================================
class start:
  def __init__(self,image=None,mode='single',port={},video=0,box=2,switch=True,**kwargs):
    fluidsynth.init('{0}/library/expokoi.sf2'.format(os.path.dirname(__file__)))
    
    instrument = kwargs.get('instrument','piano')

    if   instrument=='piano':   fluidsynth.set_instrument(0,1)
    elif instrument=='bells':   fluidsynth.set_instrument(0,11)
    elif instrument=='organ':   fluidsynth.set_instrument(0,16)
    elif instrument=='horn':    fluidsynth.set_instrument(0,60)
    elif instrument=='saw':     fluidsynth.set_instrument(0,81)
    elif instrument=='space':   fluidsynth.set_instrument(0,86)
    elif instrument=='synth':   fluidsynth.set_instrument(0,100)
    else:
      raise NotImplementedError('Instrument "{0}" not recognized'.format(instrument))
    global pressed
    global screen

    self.listener = keyboard.Listener(on_press=on_press)
    self.listener.start()  

  # Select image
  # -------------------------------------
    self.switch = switch

    if image is None:
      tkinter.Tk().withdraw()
      imgpath = filedialog.askopenfilenames()
        
      if len(imgpath)<1: sys.exit(1)
    else: imgpath = image

    if isinstance(imgpath,str): imgpath = [imgpath]

    imginit = 0

    modlist = ['single']
    if mode not in modlist:
      raise NotImplementedError('"{0}" mode is unknown'.format(mode))
    elif mode in ['scan','adaptive','party']:
      raise NotImplementedError('"{0} mode" not implemented'.format(mode))
    modinit = modlist.index(mode)

  # Start capture from webcam
  # -------------------------------------
    self.imgfull = True
    self.padfull = kwargs.get('pad',False) if self.imgfull else False

    screen = pygame.display.set_mode((scrw,scrh),pygame.FULLSCREEN)
  
    while True:
      self.opvideo = cv2.VideoCapture(video)
      self.opmusic = gethsv(imgpath[imginit])

      self.mphands = mp.solutions.hands
      self.mpdraws = mp.solutions.drawing_utils
      self.mpstyle = mp.solutions.drawing_styles

      self.opindex = 8
      self.opthumb = 4

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

    imgonly = kwargs.get('imgonly',False)

    ophands = self.mphands.Hands(max_num_hands=1)

    onmusic = False

    pxshift = kwargs.get('shift',2)
    pxshift = (pxshift/100)*np.minimum(self.opmusic.w,self.opmusic.h)

    toctime = kwargs.get('toc',0.05)
    offtime = kwargs.get('off',0.05)
    tictime = time.time()

    while True:
      _, opframe = self.opvideo.read()

      opframe = self.rescale(opframe)

      opframe = cv2.flip(opframe,1)
      imframe = cv2.cvtColor(opframe,cv2.COLOR_BGR2RGB)

      immusic = self.opmusic.bgr.copy()
      imhands = ophands.process(imframe)

      bhmidif = None
      bhmidiv = None

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

          if imlabel=='Left':
            bhmidif, bhmidiv = self.getmex(pxmusic,pxpatch,vlims,flims)
          
          if imlabel=='Right':
            rhmidif, rhmidiv = self.getmex(pxmusic,pxpatch,vlims,flims)

            bhmidif = rhmidif if bhmidif is None else int(0.50*(rhmidif+bhmidif))
            bhmidiv = rhmidiv if bhmidiv is None else int(0.50*(rhmidiv+bhmidiv))

        if (bhmidif is not None) and (bhmidiv is not None):
          if time.time()-tictime>toctime and not onmusic:
          # self.midiout.send(mido.Message('note_on',channel=8,note=bhmidif,velocity=bhmidiv))
            fluidsynth.play_Note(bhmidif,0,bhmidiv)
            pxmusicold = pxmusic
            onmusic = True

          if time.time()-tictime>toctime+offtime and np.hypot(pxmusicold[0]-pxmusic[0],pxmusicold[1]-pxmusic[1])>pxshift:
          # self.midiout.send(mido.Message('note_off',channel=8,note=bhmidif))
            self.panic(); onmusic = False
            
            tictime = time.time()
        else: self.panic()
      else: self.panic()

      mixframe = immusic

      hm, wm, _ = mixframe.shape

      if self.padfull:
        if (hm/wm)>(scrh/scrw):
          resframe = cv2.resize(mixframe,None,fx=scrh/hm,fy=scrh/hm)
          mixframe = np.zeros((scrh,scrw,mixframe.shape[2]),dtype=mixframe.dtype)
          mixframe[:,(scrw-resframe.shape[1])//2:(scrw-resframe.shape[1])//2+resframe.shape[1],:] = resframe.copy()
        else:    
          resframe = cv2.resize(mixframe,None,fx=scrw/wm,fy=scrw/wm)
          mixframe = np.zeros((scrh,scrw,mixframe.shape[2]),dtype=mixframe.dtype)
          mixframe[(scrh-resframe.shape[0])//2:(scrh-resframe.shape[0])//2+resframe.shape[0],:,:] = resframe.copy()
      else:
        if (hm/wm)>(scrh/scrw): 
          mixframe = cv2.resize(mixframe,None,fx=scrw/wm,fy=scrw/wm)
          mixframe = mixframe[(mixframe.shape[0]-scrh)//2:(mixframe.shape[0]-scrh)//2+scrh,:,:]
        else:
          mixframe = cv2.resize(mixframe,None,fx=scrh/hm,fy=scrh/hm)
          mixframe = mixframe[:,(mixframe.shape[1]-scrw)//2:(mixframe.shape[1]-scrw)//2+scrw,:]

      if not imgonly:            
        opframe  = cv2.resize(opframe,None,fx=0.20*mixframe.shape[0]/opframe.shape[0],
                                            fy=0.20*mixframe.shape[0]/opframe.shape[0])
        mixframe[                  int(0.01*mixframe.shape[0]):opframe.shape[0]+int(0.01*mixframe.shape[0]),
                  -opframe.shape[1]-int(0.01*mixframe.shape[0]):                -int(0.01*mixframe.shape[0])] = opframe

      mixframe = cv2.rotate(mixframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
      mixframe = cv2.flip(mixframe,0)
      mixframe = cv2.cvtColor(mixframe, cv2.COLOR_BGR2RGB)
      mixframe = pygame.surfarray.make_surface(mixframe)
      
      screen.blit(mixframe,(0,0))
      
      pygame.display.update()

      if (cv2.waitKey(1) & 0xFF == ord('q')) or (pressed is not None): break

    self.opvideo.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# Turn off all MIDI notes
# =====================================
  def panic(self):
    fluidsynth.stop_everything()
