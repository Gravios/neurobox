""" 
MTASession(name,varargin) 
Data structure to organize the analysis of neural and position data.

   name - string: Same name as the directory of the session

   varargin:
     [mazeName, overwrite, TTLValue, xyzSystem, ephySystem, xyzSampleRate]

     mazeName:        string, 3-4 letter name of the testing arena 
                      (e.g. 'rof' := rectangular open-field)

     overwrite:       boolean, flag to overwrite saved Sessions

     TTLValue:        string, used to synchronize position and electrophysiological data

     dataLogger:       string, name/id of the system(s) to record subject

     xyzSampleRate:   numeric, samples per second for xyz tracking
                      Vicon(M): 119.881035 Hz
                      Vicon(V): 199.997752 Hz or 149.9974321 Hz
                      Optitrack: duno

-------------------------------------------------------------------------------------------------
   General Loading:
     
     Load from saved Session,
     Session = MTASession(name,mazeName);
     
     Create new session,
     Session = MTASession(name,mazeName,overwrite,TTLValue,xyzSystem,ephySystem);

-------------------------------------------------------------------------------------------------
     examples:
       load saved session,
        Session = MTASession('jg05-20120309','rof');

       Create New Session
         Session = MTASession('jg05-20120309','rof',1,'0x0040',{'nlx','vicon'},119.881035);
   
-------------------------------------------------------------------------------------------------

        filebase - string: full file head in the format name.(Maze.name).trialName

        spath - struct: same as path but with Session name appended to the end
        
        path - struct: holds all paths of the constructed data tree created by MTAConfiguration.m
        
        name - string: name of the directory holding session information

        trialName - string: designation of trial the full Session has the default name 'all'
        
        par - struct: contains parameter information regarding the recording systems, units ect...
        
        sampleRate - double: Sample Rate of electrophysiological recording system
        
        Maze - MTAMaze: Object containing all maze information

        Model - MTAModel: Object contianing all marker information

        sync - MTADepoch: Loading periods relative to the primary recording system

        xyz - MTADxyz: (Time,Marker,Dimension) XYZ position of each marker 

        ang - MTADang: (Time,Marker,Marker,Dimension) marker to marker angles 

        stc - MTAStateCollection: Stores the periods of events
        
        spk - MTASpk: Stores neurons' action potential timing and features

        Fet - MTAFet: Object containing behavioral features

        lfp - MTADlfp: (Time,channel) local field potential

        ufr - MTADufr: (Time,ClusterId) unit firing rates with lfpSampleRate

        nq - struct: clustering quality and spike waveform characteristics

        fbr - MTADfbr: (Time, frequency) fiber photometery with lfpSampleRate
        
        arena - MTADrbo: (Time,rbo,dims) Position and Quaternion of arena

        rbo - MTADrbo: (Time,rbo,dims) Position Quaternion of subject
        
        meta - struct: contains analysis information regarding probes and subjects

"""


from neurobox.dtype import Struct
from neurobox.utils.sync import *
import numpy as np
import pickle
from dotenv import dotenv_values
from pathlib import Path

#from neurobox.dtype import Arena
#from neurobox.dtype import Trial


class NBSession:
    

    def __init__(self, session_name, maze_name:str="cof", overwrite:bool=False, data_loggers:list[str]=["nlx","vicon"], TTLValue:str="0x0040"):
        
        project_conf = dotenv_values()

        project_name  = project_conf["NB_PROJECT_NAME"]
        
        NB_PROJECT_PATH  = Path(project_conf["NB_PROJECT_PATH"]) / project_name
        NB_PROJECT_CONFIG = NB_PROJECT_PATH / "config"
        
        paths = np.load( NB_PROJECT_CONFIG / "NBpaths.npy", allow_pickle=True)

        paths = paths[()]

        self.path = Struct( paths )

        # !!! COPY the object - needs recursive method
        if isinstance(session_name, NBSession):
            for key, value in self.__dict__.items():
                self.__setattr__[key] = value
                
        elif isinstance(session_name, str):
            # CREATE new session 
            self.name:str = session_name
            self.spath:Path = self.path.project / self.name
            self.trial:str = "all"
            self.maze:str = "cof"
            self.filebase:str = ".".join([self.name, self.maze, self.trial])
            if not overwrite and Path(self.spath / f"{self.filebase}.ses.pkl").is_file():
                self.load()
                self.update_paths()
            else:                
                self.create(data_loggers=data_loggers, TTLValue=TTLValue)
                
        elif session_name is None:
            raise ValueError("neurobox.dtypes.session: session_name is not set")
        
        else:
            raise ValueError("neurobox.dtypes.session: session initialization faild.")
                             

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        
    def peek(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        
    @staticmethod
    def validate(session):
        if isinstance(session_name, MTATrial):
            Session = MTASession.validate(Session.filebase)
        elif isinstance(session_name, MTASession):
            return
        elif isinstance(Session, str):
            pass
        elif isinstance(Session, Struct):
            pass
        else:
            #error('MTA:validate_trial: unrecognized format');
            pass
        
    def create(self, data_loggers:list[str], TTLValue:str):
        dl = "_".join(data_loggers)
        globals()[f"sync_{dl}"](TTLValue)

    def load(self):
        fpkl = Path(self.spath / f"{self.filebase}.ses.pkl")
        buffers = []
        with open( fpkl, 'rb') as inp:
            pkcnt = pickle.load(inp, buffers=buffers)
            self.__dict__.update(pkcnt)
    
    def save(self):
        with open(Path(self.spath / f"{self.filebase}.ses.pkl"), 'wb') as outp:
            pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)

