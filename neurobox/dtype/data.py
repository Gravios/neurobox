"""
MTAData(path,filename,data,sampleRate,syncPeriods,syncOrigin,type,ext)

  MTAData is a superclass of most MTA data types. It is a container for 
  general data types, which alows dynamic refrencing and the use of a set
  of common functions.

  Current Data Types: 
    TimeSeries, TimePeriods, TimePoints, SpatialMap

  Current Subclasses:
    MTADang, MTADepoch, MTADlfp, MTADufr, MTADxyz

  Indexing (TimeSeries):
    first dimension:    time, ':', numeric array of indicies or
                              start and stop periods in an nx2 matrix 

    second dimension:   channel/marker, ':', numeric array of indicies or
                              string corresponding to one of the model labels
    
    Nth dimension:      subspace/channel/marker, :', numeric array of
                              indicies or string corresponding to one of 
                              the model labels

    Indexing Example:
       MTADxyz TimeSeries, xy coordinates of 2 markers for all time
       xy_head = xyz(:,{'head_back','head_front'},[1,2]);

       MTADxyz TimeSeries, z coordinates of 2 markers for specific periods
       z_head = xyz([1,300;400,1000],'head_front',3);

       MTADang TimeSeries, pitch of 2 markers for all time
       spine_pitch = ang(:,'spine_middle','spine_upper',2);

  varargin:
    
    path:       string, the directory where the object's data is stored

    filename:   string, The file name of the .mat file which contains the 
                        objects data

    data:       matrix, Data is your data, ... so put your data here

    sampleRate: double, Sampling rate of the associated data
        
    syncPeriods: 
           MTADepoch,    Time in seconds or the indicies indicating where
                         the data fits in the Session
           numericArray, The absolute Recording indicies

    syncOrigin: double, Time or index where the data exits in the overall
                        session

    type:       string, A short string which denotes the type of data held
                        by the object

    ext:        string, A short, unique string which will be the primary
                        file identifier

    sampleRate: double, The sampling rate of the associated data
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class NBData(ABC):

    def __init__(self, path:Path, filename:str, data:np.ndarray, samplerate:np.float64,
                 syncperiods:NBEpoch, syncorigin:np.float64, dclass:str,
                 ext:str, name:str, label:str, key:str):
        self.filename = filename
        self.path = path
        self.data = data
        self.class = dclass
        self.ext = ext
        self.name = name
        self.label = label
        self.key = key
        self.samplerate = samplerate
        self.sync = syncperiods
        self.origin = syncorigin
        self._hash = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
        self.hash()

    @abstractmethod
    def create(self) -> None:
        pass

    @abstractmethod
    def update_hash(self) -> None:
        pass

    def __hash__(self):
        self._hash = hash((self.filename, self.data, self.class))
        return self.hash
        
    def __eq__(self, other):
        if  isinstance(other, NBData):
            return self.data != other.data
        else:
            return self.data != other
        
    def __eq__(self, other):
        if  isinstance(other, NBData):
            return self.data == other.data
        else:
            return self.data == other

    def __gt__(self, other):
        if  isinstance(other, NBData):
            return self.data > other.data
        else:
            return self.data > other
        
    def __ge__(self, other):
        if  isinstance(other, NBData):
            return self.data >= other.data
        else:
            return self.data >= other

    def __lt__(self, other):
        if  isinstance(other, NBData):
            return self.data < other.data
        else:
            return self.data < other
        
    def __lt__(self, other):
        if  isinstance(other, NBData):
            return self.data <= other.data
        else:
            return self.data <= other

    

class NBDxyz(NBData):
    pass



