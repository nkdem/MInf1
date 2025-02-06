# Data Processing
After downloading the raw cuts of HEAR-DS data from [here](https://download.hz-ol.de/hear-ds-data/HEAR-DS/RawAudioCuts/) 
and unzipping them accordingly, you will have a directory structure like this:
```
CocktailParty/
InTraffic/
InVehicle/
Music/
QuietIndoors/
ReverberantEnvironment/
WindTurbulence/
```

Each of these environments will have at least three recording situations (RECSITs) so each folder will have at least three subfolders with the following naming convention:
```
rec_id_<REC_ID>_cut_<CUT_I>_<DESCRIPTION>_<TRACKNAME>_<EXPORTFORMAT>.wav
```
where `<REC_ID>` is a 3 digit number, `<CUT_I>` is a 2 digit number, `<DESCRIPTION>` is a description of the sound event, `<TRACKNAME>` is the name of the microphone used, and `<EXPORTFORMAT>` is the name of the audio-exporter used.

An example of `ls CocktailParty/rec_001_HDH_1`:
```
rec_id_001_cut_00_multispeaker_00_Mic_BTE_R_rear_raw_48kHz32bit.wav
rec_id_001_cut_00_multispeaker_00_Mic_BTE_R_front_raw_48kHz32bit.wav
rec_id_001_cut_00_multispeaker_00_Mic_BTE_L_rear_raw_48kHz32bit.wav
rec_id_001_cut_00_multispeaker_00_Mic_BTE_L_front_raw_48kHz32bit.wav
rec_id_001_cut_00_multispeaker_00_Mic_ITC_R_raw_48kHz32bit.wav
rec_id_001_cut_00_multispeaker_00_Mic_ITC_L_raw_48kHz32bit.wav
```
So a single audio cut from a recording situation will have 6 audio files, 3 for each ear (L and R) and for each microphone (BTE Rear and Front, or ITC).

The music environment contains .wav files in 16 bits or 32 bits, so there would be a total of 12 files for each audio cut. 


