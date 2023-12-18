# asa_challenge
ASA Challenge - 2023 First Place Winner with Ben Masters

Check the `final_report.pdf` for our submission and approach.


## Underwater Audio Processing, hydrophones N, O, P


### Task 1: examine spectrogram, estimate diver breathing rate in Hz

Approach:
1) Generate spectrogram
2) Examine noticeable patterns, comment on spectral properties of diver's acoustic signature

3) Estimate breathing rate:
- find changes in spectrogram in time that corresponds with breathing, compute breathing rate
- compute time between peaks in frame based power, indicating where breaths are taking place, to generate a breathing rate


### Task 2: find time of closest approach to hydrophone O, estimate diver's altitude, swimming speed

Approach:
- find where energy is maximal at hydrophone O, indicating that diver is closest (assuming certain conditions, like the breaths are at a similar level throughout w.r.t the diver)
- corroborate this time with the audio direction of arrival at hydrophone O using all 3 sensors:

1) estimate frame-to-frame timing differences with spectral beamform approach
2) estimate DOA with a trigonometric solution
3) ensemble results from sensors N-O, N-P and O-P
