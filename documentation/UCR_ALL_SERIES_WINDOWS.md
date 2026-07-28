# Tutte le 250 serie UCR — uso, finestre per client, adeguatezza del window=128

`uso`: **run** = nel run in corso (82) · **ds** = nel dataset ma non nel run (144) · **NO** = scartata (24, train troppo corto)

`fin@128` = finestre stride-1 del train CONTIGUO. `10%/20%/30%` = finestre per singolo client
(ce ne sono **due** al 10% e **due** al 20%; differiscono al massimo di 1 finestra).
Per le serie `NO` i valori sono ipotetici, calcolati con la stessa regola di partizione.

`cicli/2.0` = quanti cicli del segnale la nostra finestra da 128 contiene (`128/periodo`).
**Il paper ne vuole 2.0** (`T = 2×periodo`). `⚠2P` = passando a 2P questo cluster perderebbe almeno un client per mancanza di finestre.

| # | nome serie UCR | uso | train | fin@128 | 10% | 20% | 30% | per. | cicli/2.0 | 2P | verdetto sul 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | DISTORTED1sddb40 | ds | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 204 | 0.63 | 408 | stretta |
| 002 | DISTORTED2sddb40 | ds | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 204 | 0.63 | 408 | stretta |
| 003 | DISTORTED3sddb40 | ds | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 204 | 0.63 | 408 | stretta |
| 004 | DISTORTEDBIDMC1 | NO | 2,500 | 2,123 | 123 | 373 | 623 | 82 | 1.56 | 164 | un po' stretta |
| 005 | DISTORTEDCIMIS44AirTempera | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 006 | DISTORTEDCIMIS44AirTempera | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 007 | DISTORTEDCIMIS44AirTempera | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 008 | DISTORTEDCIMIS44AirTempera | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 009 | DISTORTEDCIMIS44AirTempera | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 010 | DISTORTEDCIMIS44AirTempera | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 011 | DISTORTEDECG1 | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 91 | 1.41 | 182 | un po' stretta |
| 012 | DISTORTEDECG2 | ds | 15,000 | 13,373 | 1,373 | 2,873 | 4,373 | 93 | 1.38 | 186 | un po' stretta |
| 013 | DISTORTEDECG3 | ds | 15,000 | 13,373 | 1,373 | 2,873 | 4,373 | 167 | 0.77 | 334 | stretta |
| 014 | DISTORTEDECG3 | ds | 8,000 | 7,073 | 673 | 1,473 | 2,273 | 165 | 0.78 | 330 | stretta |
| 015 | DISTORTEDECG4 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 479 | 0.27 | 958 | MOLTO stretta ⚠2P |
| 016 | DISTORTEDECG4 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 166 | 0.77 | 332 | stretta |
| 017 | DISTORTEDECG4 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 166 | 0.77 | 332 | stretta |
| 018 | DISTORTEDECG4 | ds | 8,000 | 7,073 | 673 | 1,473 | 2,273 | 168 | 0.76 | 336 | stretta |
| 019 | DISTORTEDGP711MarkerLFM5z1 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 020 | DISTORTEDGP711MarkerLFM5z2 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 021 | DISTORTEDGP711MarkerLFM5z3 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 022 | DISTORTEDGP711MarkerLFM5z4 | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 215 | 0.60 | 430 | stretta ⚠2P |
| 023 | DISTORTEDGP711MarkerLFM5z5 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 217 | 0.59 | 434 | stretta |
| 024 | DISTORTEDInternalBleeding1 | ds | 3,200 | 2,753 | 193 | 513 | 833 | 153 | 0.84 | 306 | stretta |
| 025 | DISTORTEDInternalBleeding1 | ds | 2,800 | 2,393 | 153 | 433 | 713 | 152 | 0.84 | 304 | stretta ⚠2P |
| 026 | DISTORTEDInternalBleeding1 | NO | 1,700 | 1,403 | 43 | 213 | 383 | 174 | 0.74 | 348 | stretta |
| 027 | DISTORTEDInternalBleeding1 | NO | 1,200 | 953 | 0 | 113 | 233 | 184 | 0.70 | 368 | stretta |
| 028 | DISTORTEDInternalBleeding1 | NO | 1,600 | 1,313 | 33 | 193 | 353 | 181 | 0.71 | 362 | stretta |
| 029 | DISTORTEDInternalBleeding1 | NO | 2,300 | 1,943 | 103 | 333 | 563 | 183 | 0.70 | 366 | stretta |
| 030 | DISTORTEDInternalBleeding1 | ds | 3,000 | 2,573 | 173 | 473 | 773 | 183 | 0.70 | 366 | stretta ⚠2P |
| 031 | DISTORTEDInternalBleeding2 | ds | 2,700 | 2,303 | 143 | 413 | 683 | 183 | 0.70 | 366 | stretta ⚠2P |
| 032 | DISTORTEDInternalBleeding4 | NO | 1,000 | 773 | 0 | 73 | 173 | 180 | 0.71 | 360 | stretta |
| 033 | DISTORTEDInternalBleeding5 | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 175 | 0.73 | 350 | stretta |
| 034 | DISTORTEDInternalBleeding6 | NO | 1,500 | 1,223 | 23 | 173 | 323 | 153 | 0.84 | 306 | stretta |
| 035 | DISTORTEDInternalBleeding8 | NO | 2,500 | 2,123 | 123 | 373 | 623 | 165 | 0.78 | 330 | stretta |
| 036 | DISTORTEDInternalBleeding9 | ds | 4,200 | 3,653 | 293 | 713 | 1,133 | 175 | 0.73 | 350 | stretta |
| 037 | DISTORTEDLab2Cmac011215EPG | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 35 | 3.66 | 70 | troppo larga |
| 038 | DISTORTEDLab2Cmac011215EPG | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 35 | 3.66 | 70 | troppo larga |
| 039 | DISTORTEDLab2Cmac011215EPG | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 35 | 3.66 | 70 | troppo larga |
| 040 | DISTORTEDLab2Cmac011215EPG | ds | 6,000 | 5,273 | 473 | 1,073 | 1,673 | 32 | 4.00 | 64 | troppo larga |
| 041 | DISTORTEDLab2Cmac011215EPG | ds | 7,000 | 6,173 | 573 | 1,273 | 1,973 | 32 | 4.00 | 64 | troppo larga |
| 042 | DISTORTEDLab2Cmac011215EPG | ds | 7,000 | 6,173 | 573 | 1,273 | 1,973 | 32 | 4.00 | 64 | troppo larga |
| 043 | DISTORTEDMesoplodonDensiro | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 207 | 0.62 | 414 | stretta |
| 044 | DISTORTEDPowerDemand1 | ds | 9,000 | 7,973 | 773 | 1,673 | 2,573 | 24 | 5.33 | 48 | troppo larga |
| 045 | DISTORTEDPowerDemand2 | ds | 14,000 | 12,473 | 1,273 | 2,673 | 4,073 | 24 | 5.33 | 48 | troppo larga |
| 046 | DISTORTEDPowerDemand3 | ds | 16,000 | 14,273 | 1,473 | 3,073 | 4,673 | 24 | 5.33 | 48 | troppo larga |
| 047 | DISTORTEDPowerDemand4 | ds | 18,000 | 16,073 | 1,673 | 3,473 | 5,273 | 24 | 5.33 | 48 | troppo larga |
| 048 | DISTORTEDTkeepFifthMARS | ds | 3,500 | 3,023 | 223 | 573 | 923 | 97 | 1.32 | 194 | un po' stretta |
| 049 | DISTORTEDTkeepFirstMARS | ds | 3,500 | 3,023 | 223 | 573 | 923 | 97 | 1.32 | 194 | un po' stretta |
| 050 | DISTORTEDTkeepForthMARS | ds | 3,500 | 3,023 | 223 | 573 | 923 | 97 | 1.32 | 194 | un po' stretta |
| 051 | DISTORTEDTkeepSecondMARS | ds | 3,500 | 3,023 | 223 | 573 | 923 | 97 | 1.32 | 194 | un po' stretta |
| 052 | DISTORTEDTkeepThirdMARS | ds | 3,500 | 3,023 | 223 | 573 | 923 | 97 | 1.32 | 194 | un po' stretta |
| 053 | DISTORTEDWalkingAceleratio | NO | 1,500 | 1,223 | 23 | 173 | 323 | 114 | 1.12 | 228 | un po' stretta |
| 054 | DISTORTEDWalkingAceleratio | ds | 2,700 | 2,303 | 143 | 413 | 683 | 114 | 1.12 | 228 | un po' stretta |
| 055 | DISTORTEDapneaecg2 | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 94 | 1.36 | 188 | un po' stretta |
| 056 | DISTORTEDapneaecg3 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 94 | 1.36 | 188 | un po' stretta |
| 057 | DISTORTEDapneaecg4 | ds | 6,000 | 5,273 | 473 | 1,073 | 1,673 | 94 | 1.36 | 188 | un po' stretta |
| 058 | DISTORTEDapneaecg | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 96 | 1.33 | 192 | un po' stretta |
| 059 | DISTORTEDgait1 | ds | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 328 | 0.39 | 656 | MOLTO stretta |
| 060 | DISTORTEDgait2 | ds | 22,000 | 19,673 | 2,073 | 4,273 | 6,473 | 328 | 0.39 | 656 | MOLTO stretta |
| 061 | DISTORTEDgait3 | ds | 24,500 | 21,923 | 2,323 | 4,773 | 7,223 | 328 | 0.39 | 656 | MOLTO stretta |
| 062 | DISTORTEDgaitHunt1 | ds | 18,500 | 16,523 | 1,723 | 3,573 | 5,423 | 328 | 0.39 | 656 | MOLTO stretta |
| 063 | DISTORTEDgaitHunt2 | ds | 18,500 | 16,523 | 1,723 | 3,573 | 5,423 | 328 | 0.39 | 656 | MOLTO stretta |
| 064 | DISTORTEDgaitHunt3 | ds | 23,400 | 20,933 | 2,213 | 4,553 | 6,893 | 328 | 0.39 | 656 | MOLTO stretta |
| 065 | DISTORTEDinsectEPG1 | ds | 3,000 | 2,573 | 173 | 473 | 773 | 25 | 5.12 | 50 | troppo larga |
| 066 | DISTORTEDinsectEPG2 | ds | 3,700 | 3,203 | 243 | 613 | 983 | 25 | 5.12 | 50 | troppo larga |
| 067 | DISTORTEDinsectEPG3 | ds | 5,200 | 4,553 | 393 | 913 | 1,433 | 25 | 5.12 | 50 | troppo larga |
| 068 | DISTORTEDinsectEPG4 | NO | 1,300 | 1,043 | 3 | 133 | 263 | 25 | 5.12 | 50 | troppo larga |
| 069 | DISTORTEDinsectEPG5 | ds | 3,200 | 2,753 | 193 | 513 | 833 | 25 | 5.12 | 50 | troppo larga |
| 070 | DISTORTEDltstdbs30791AI | ds | 17,555 | 15,672 | 1,628 | 3,384 | 5,139 | 221 | 0.58 | 442 | stretta |
| 071 | DISTORTEDltstdbs30791AS | ds | 23,000 | 20,573 | 2,173 | 4,473 | 6,773 | 221 | 0.58 | 442 | stretta |
| 072 | DISTORTEDltstdbs30791ES | ds | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 228 | 0.56 | 456 | stretta |
| 073 | DISTORTEDpark3m | ds | 60,000 | 53,873 | 5,873 | 11,873 | 17,873 | 356 | 0.36 | 712 | MOLTO stretta |
| 074 | DISTORTEDqtdbSel1005V | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 206 | 0.62 | 412 | stretta ⚠2P |
| 075 | DISTORTEDqtdbSel100MLII | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 206 | 0.62 | 412 | stretta ⚠2P |
| 076 | DISTORTEDresperation10 | ds | 48,000 | 43,073 | 4,673 | 9,473 | 14,273 | 459 | 0.28 | 918 | MOLTO stretta |
| 077 | DISTORTEDresperation11 | ds | 58,000 | 52,073 | 5,673 | 11,473 | 17,273 | 462 | 0.28 | 924 | MOLTO stretta |
| 078 | DISTORTEDresperation1 | ds | 100,000 | 89,873 | 9,873 | 19,873 | 29,873 | 478 | 0.27 | 956 | MOLTO stretta |
| 079 | DISTORTEDresperation2 | ds | 30,000 | 26,873 | 2,873 | 5,873 | 8,873 | 478 | 0.27 | 956 | MOLTO stretta |
| 080 | DISTORTEDresperation2 | ds | 30,000 | 26,873 | 2,873 | 5,873 | 8,873 | 478 | 0.27 | 956 | MOLTO stretta |
| 081 | DISTORTEDresperation3 | ds | 45,000 | 40,373 | 4,373 | 8,873 | 13,373 | 478 | 0.27 | 956 | MOLTO stretta |
| 082 | DISTORTEDresperation4 | ds | 70,000 | 62,873 | 6,873 | 13,873 | 20,873 | 446 | 0.29 | 892 | MOLTO stretta |
| 083 | DISTORTEDresperation9 | ds | 38,000 | 34,073 | 3,673 | 7,473 | 11,273 | 485 | 0.26 | 970 | MOLTO stretta |
| 084 | DISTORTEDs20101mML2 | ds | 12,000 | 10,673 | 1,073 | 2,273 | 3,473 | 100 | 1.28 | 200 | un po' stretta |
| 085 | DISTORTEDs20101m | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 100 | 1.28 | 200 | un po' stretta |
| 086 | DISTORTEDsddb49 | ds | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 268 | 0.48 | 536 | MOLTO stretta |
| 087 | DISTORTEDsel840mECG1 | ds | 17,000 | 15,173 | 1,573 | 3,273 | 4,973 | 172 | 0.74 | 344 | stretta |
| 088 | DISTORTEDsel840mECG2 | ds | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 170 | 0.75 | 340 | stretta |
| 089 | DISTORTEDtiltAPB1 | ds | 100,000 | 89,873 | 9,873 | 19,873 | 29,873 | 250 | 0.51 | 500 | stretta |
| 090 | DISTORTEDtiltAPB2 | ds | 50,000 | 44,873 | 4,873 | 9,873 | 14,873 | 250 | 0.51 | 500 | stretta |
| 091 | DISTORTEDtiltAPB3 | ds | 40,000 | 35,873 | 3,873 | 7,873 | 11,873 | 250 | 0.51 | 500 | stretta |
| 092 | DISTORTEDtiltAPB4 | ds | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 250 | 0.51 | 500 | stretta |
| 093 | NOISE1sddb40 | ds | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 214 | 0.60 | 428 | stretta |
| 094 | NOISEBIDMC1 | NO | 2,500 | 2,123 | 123 | 373 | 623 | 81 | 1.58 | 162 | un po' stretta |
| 095 | NOISECIMIS44AirTemperature | ds | 4,000 | 3,473 | 273 | 673 | 1,073 | 24 | 5.33 | 48 | troppo larga |
| 096 | NOISEECG4 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 167 | 0.77 | 334 | stretta |
| 097 | NOISEGP711MarkerLFM5z3 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 098 | NOISEInternalBleeding16 | NO | 1,200 | 953 | 0 | 113 | 233 | 181 | 0.71 | 362 | stretta |
| 099 | NOISEInternalBleeding6 | NO | 1,500 | 1,223 | 23 | 173 | 323 | 150 | 0.85 | 300 | stretta |
| 100 | NOISELab2Cmac011215EPG1 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 34 | 3.76 | 68 | troppo larga |
| 101 | NOISELab2Cmac011215EPG4 | ds | 6,000 | 5,273 | 473 | 1,073 | 1,673 | 34 | 3.76 | 68 | troppo larga |
| 102 | NOISEMesoplodonDensirostri | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 204 | 0.63 | 408 | stretta |
| 103 | NOISETkeepThirdMARS | ds | 3,500 | 3,023 | 223 | 573 | 923 | 98 | 1.31 | 196 | un po' stretta |
| 104 | NOISEapneaecg4 | ds | 6,000 | 5,273 | 473 | 1,073 | 1,673 | 96 | 1.33 | 192 | un po' stretta |
| 105 | NOISEgait3 | ds | 24,500 | 21,923 | 2,323 | 4,773 | 7,223 | 323 | 0.40 | 646 | MOLTO stretta |
| 106 | NOISEgaitHunt2 | ds | 18,500 | 16,523 | 1,723 | 3,573 | 5,423 | 350 | 0.37 | 700 | MOLTO stretta |
| 107 | NOISEinsectEPG3 | ds | 5,200 | 4,553 | 393 | 913 | 1,433 | 25 | 5.12 | 50 | troppo larga |
| 108 | NOISEresperation2 | ds | 30,000 | 26,873 | 2,873 | 5,873 | 8,873 | 464 | 0.28 | 928 | MOLTO stretta |
| 109 | 1sddb40 | run | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 214 | 0.60 | 428 | stretta |
| 110 | 2sddb40 | run | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 214 | 0.60 | 428 | stretta |
| 111 | 3sddb40 | run | 35,000 | 31,373 | 3,373 | 6,873 | 10,373 | 214 | 0.60 | 428 | stretta |
| 112 | BIDMC1 | NO | 2,500 | 2,123 | 123 | 373 | 623 | 81 | 1.58 | 162 | un po' stretta |
| 113 | CIMIS44AirTemperature1 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 23 | 5.57 | 46 | troppo larga |
| 114 | CIMIS44AirTemperature2 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 23 | 5.57 | 46 | troppo larga |
| 115 | CIMIS44AirTemperature3 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 23 | 5.57 | 46 | troppo larga |
| 116 | CIMIS44AirTemperature4 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 23 | 5.57 | 46 | troppo larga |
| 117 | CIMIS44AirTemperature5 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 23 | 5.57 | 46 | troppo larga |
| 118 | CIMIS44AirTemperature6 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 23 | 5.57 | 46 | troppo larga |
| 119 | ECG1 | run | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 94 | 1.36 | 188 | un po' stretta |
| 120 | ECG2 | run | 15,000 | 13,373 | 1,373 | 2,873 | 4,373 | 94 | 1.36 | 188 | un po' stretta |
| 121 | ECG3 | ds | 15,000 | 13,373 | 1,373 | 2,873 | 4,373 | 164 | 0.78 | 328 | stretta |
| 122 | ECG3 | run | 8,000 | 7,073 | 673 | 1,473 | 2,273 | 164 | 0.78 | 328 | stretta |
| 123 | ECG4 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 475 | 0.27 | 950 | MOLTO stretta ⚠2P |
| 124 | ECG4 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 166 | 0.77 | 332 | stretta |
| 125 | ECG4 | ds | 5,000 | 4,373 | 373 | 873 | 1,373 | 166 | 0.77 | 332 | stretta |
| 126 | ECG4 | ds | 8,000 | 7,073 | 673 | 1,473 | 2,273 | 166 | 0.77 | 332 | stretta |
| 127 | GP711MarkerLFM5z1 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 128 | GP711MarkerLFM5z2 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 129 | GP711MarkerLFM5z3 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 130 | GP711MarkerLFM5z4 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 219 | 0.58 | 438 | stretta ⚠2P |
| 131 | GP711MarkerLFM5z5 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 219 | 0.58 | 438 | stretta |
| 132 | InternalBleeding10 | run | 3,200 | 2,753 | 193 | 513 | 833 | 155 | 0.83 | 310 | stretta |
| 133 | InternalBleeding14 | run | 2,800 | 2,393 | 153 | 433 | 713 | 155 | 0.83 | 310 | stretta ⚠2P |
| 134 | InternalBleeding15 | NO | 1,700 | 1,403 | 43 | 213 | 383 | 175 | 0.73 | 350 | stretta |
| 135 | InternalBleeding16 | NO | 1,200 | 953 | 0 | 113 | 233 | 182 | 0.70 | 364 | stretta |
| 136 | InternalBleeding17 | NO | 1,600 | 1,313 | 33 | 193 | 353 | 182 | 0.70 | 364 | stretta |
| 137 | InternalBleeding18 | NO | 2,300 | 1,943 | 103 | 333 | 563 | 182 | 0.70 | 364 | stretta |
| 138 | InternalBleeding19 | run | 3,000 | 2,573 | 173 | 473 | 773 | 182 | 0.70 | 364 | stretta ⚠2P |
| 139 | InternalBleeding20 | run | 2,700 | 2,303 | 143 | 413 | 683 | 182 | 0.70 | 364 | stretta ⚠2P |
| 140 | InternalBleeding4 | NO | 1,000 | 773 | 0 | 73 | 173 | 180 | 0.71 | 360 | stretta |
| 141 | InternalBleeding5 | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 177 | 0.72 | 354 | stretta |
| 142 | InternalBleeding6 | NO | 1,500 | 1,223 | 23 | 173 | 323 | 152 | 0.84 | 304 | stretta |
| 143 | InternalBleeding8 | NO | 2,500 | 2,123 | 123 | 373 | 623 | 167 | 0.77 | 334 | stretta |
| 144 | InternalBleeding9 | run | 4,200 | 3,653 | 293 | 713 | 1,133 | 175 | 0.73 | 350 | stretta |
| 145 | Lab2Cmac011215EPG1 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 38 | 3.37 | 76 | troppo larga |
| 146 | Lab2Cmac011215EPG2 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 38 | 3.37 | 76 | troppo larga |
| 147 | Lab2Cmac011215EPG3 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 38 | 3.37 | 76 | troppo larga |
| 148 | Lab2Cmac011215EPG4 | run | 6,000 | 5,273 | 473 | 1,073 | 1,673 | 38 | 3.37 | 76 | troppo larga |
| 149 | Lab2Cmac011215EPG5 | run | 7,000 | 6,173 | 573 | 1,273 | 1,973 | 38 | 3.37 | 76 | troppo larga |
| 150 | Lab2Cmac011215EPG6 | run | 7,000 | 6,173 | 573 | 1,273 | 1,973 | 38 | 3.37 | 76 | troppo larga |
| 151 | MesoplodonDensirostris | run | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 203 | 0.63 | 406 | stretta |
| 152 | PowerDemand1 | run | 9,000 | 7,973 | 773 | 1,673 | 2,573 | 25 | 5.12 | 50 | troppo larga |
| 153 | PowerDemand2 | run | 14,000 | 12,473 | 1,273 | 2,673 | 4,073 | 25 | 5.12 | 50 | troppo larga |
| 154 | PowerDemand3 | run | 16,000 | 14,273 | 1,473 | 3,073 | 4,673 | 25 | 5.12 | 50 | troppo larga |
| 155 | PowerDemand4 | run | 18,000 | 16,073 | 1,673 | 3,473 | 5,273 | 25 | 5.12 | 50 | troppo larga |
| 156 | TkeepFifthMARS | run | 3,500 | 3,023 | 223 | 573 | 923 | 98 | 1.31 | 196 | un po' stretta |
| 157 | TkeepFirstMARS | run | 3,500 | 3,023 | 223 | 573 | 923 | 98 | 1.31 | 196 | un po' stretta |
| 158 | TkeepForthMARS | run | 3,500 | 3,023 | 223 | 573 | 923 | 98 | 1.31 | 196 | un po' stretta |
| 159 | TkeepSecondMARS | run | 3,500 | 3,023 | 223 | 573 | 923 | 98 | 1.31 | 196 | un po' stretta |
| 160 | TkeepThirdMARS | run | 3,500 | 3,023 | 223 | 573 | 923 | 98 | 1.31 | 196 | un po' stretta |
| 161 | WalkingAceleration1 | NO | 1,500 | 1,223 | 23 | 173 | 323 | 113 | 1.13 | 226 | un po' stretta |
| 162 | WalkingAceleration5 | run | 2,700 | 2,303 | 143 | 413 | 683 | 113 | 1.13 | 226 | un po' stretta |
| 163 | apneaecg2 | run | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 95 | 1.35 | 190 | un po' stretta |
| 164 | apneaecg3 | run | 5,000 | 4,373 | 373 | 873 | 1,373 | 95 | 1.35 | 190 | un po' stretta |
| 165 | apneaecg4 | run | 6,000 | 5,273 | 473 | 1,073 | 1,673 | 95 | 1.35 | 190 | un po' stretta |
| 166 | apneaecg | run | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 95 | 1.35 | 190 | un po' stretta |
| 167 | gait1 | run | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 319 | 0.40 | 638 | MOLTO stretta |
| 168 | gait2 | run | 22,000 | 19,673 | 2,073 | 4,273 | 6,473 | 319 | 0.40 | 638 | MOLTO stretta |
| 169 | gait3 | run | 24,500 | 21,923 | 2,323 | 4,773 | 7,223 | 319 | 0.40 | 638 | MOLTO stretta |
| 170 | gaitHunt1 | run | 18,500 | 16,523 | 1,723 | 3,573 | 5,423 | 337 | 0.38 | 674 | MOLTO stretta |
| 171 | gaitHunt2 | run | 18,500 | 16,523 | 1,723 | 3,573 | 5,423 | 337 | 0.38 | 674 | MOLTO stretta |
| 172 | gaitHunt3 | run | 23,400 | 20,933 | 2,213 | 4,553 | 6,893 | 337 | 0.38 | 674 | MOLTO stretta |
| 173 | insectEPG1 | run | 3,000 | 2,573 | 173 | 473 | 773 | 26 | 4.92 | 52 | troppo larga |
| 174 | insectEPG2 | run | 3,700 | 3,203 | 243 | 613 | 983 | 26 | 4.92 | 52 | troppo larga |
| 175 | insectEPG3 | run | 5,200 | 4,553 | 393 | 913 | 1,433 | 26 | 4.92 | 52 | troppo larga |
| 176 | insectEPG4 | NO | 1,300 | 1,043 | 3 | 133 | 263 | 26 | 4.92 | 52 | troppo larga |
| 177 | insectEPG5 | run | 3,200 | 2,753 | 193 | 513 | 833 | 26 | 4.92 | 52 | troppo larga |
| 178 | ltstdbs30791AI | run | 17,555 | 15,672 | 1,628 | 3,384 | 5,139 | 224 | 0.57 | 448 | stretta |
| 179 | ltstdbs30791AS | run | 23,000 | 20,573 | 2,173 | 4,473 | 6,773 | 224 | 0.57 | 448 | stretta |
| 180 | ltstdbs30791ES | run | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 223 | 0.57 | 446 | stretta |
| 181 | park3m | ds | 60,000 | 53,873 | 5,873 | 11,873 | 17,873 | 355 | 0.36 | 710 | MOLTO stretta |
| 182 | qtdbSel1005V | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 205 | 0.62 | 410 | stretta ⚠2P |
| 183 | qtdbSel100MLII | run | 4,000 | 3,473 | 273 | 673 | 1,073 | 183 | 0.70 | 366 | stretta |
| 184 | resperation10 | run | 48,000 | 43,073 | 4,673 | 9,473 | 14,273 | 470 | 0.27 | 940 | MOLTO stretta |
| 185 | resperation11 | ds | 58,000 | 52,073 | 5,673 | 11,473 | 17,273 | 475 | 0.27 | 950 | MOLTO stretta |
| 186 | resperation1 | ds | 100,000 | 89,873 | 9,873 | 19,873 | 29,873 | 475 | 0.27 | 950 | MOLTO stretta |
| 187 | resperation2 | ds | 30,000 | 26,873 | 2,873 | 5,873 | 8,873 | 475 | 0.27 | 950 | MOLTO stretta |
| 188 | resperation2 | run | 30,000 | 26,873 | 2,873 | 5,873 | 8,873 | 475 | 0.27 | 950 | MOLTO stretta |
| 189 | resperation3 | run | 45,000 | 40,373 | 4,373 | 8,873 | 13,373 | 475 | 0.27 | 950 | MOLTO stretta |
| 190 | resperation4 | ds | 70,000 | 62,873 | 6,873 | 13,873 | 20,873 | 475 | 0.27 | 950 | MOLTO stretta |
| 191 | resperation9 | run | 38,000 | 34,073 | 3,673 | 7,473 | 11,273 | 475 | 0.27 | 950 | MOLTO stretta |
| 192 | s20101mML2 | run | 12,000 | 10,673 | 1,073 | 2,273 | 3,473 | 99 | 1.29 | 198 | un po' stretta |
| 193 | s20101m | run | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 99 | 1.29 | 198 | un po' stretta |
| 194 | sddb49 | run | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 274 | 0.47 | 548 | MOLTO stretta |
| 195 | sel840mECG1 | run | 17,000 | 15,173 | 1,573 | 3,273 | 4,973 | 165 | 0.78 | 330 | stretta |
| 196 | sel840mECG2 | run | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 171 | 0.75 | 342 | stretta |
| 197 | tiltAPB1 | ds | 100,000 | 89,873 | 9,873 | 19,873 | 29,873 | 248 | 0.52 | 496 | stretta |
| 198 | tiltAPB2 | run | 50,000 | 44,873 | 4,873 | 9,873 | 14,873 | 248 | 0.52 | 496 | stretta |
| 199 | tiltAPB3 | run | 40,000 | 35,873 | 3,873 | 7,873 | 11,873 | 248 | 0.52 | 496 | stretta |
| 200 | tiltAPB4 | run | 20,000 | 17,873 | 1,873 | 3,873 | 5,873 | 248 | 0.52 | 496 | stretta |
| 201 | CHARISfive | ds | 10,000 | 8,873 | 873 | 1,873 | 2,873 | 32 | 4.00 | 64 | troppo larga |
| 202 | CHARISfive | ds | 10,411 | 9,242 | 914 | 1,955 | 2,996 | 31 | 4.13 | 62 | troppo larga |
| 203 | CHARISfive | ds | 10,500 | 9,323 | 923 | 1,973 | 3,023 | 32 | 4.00 | 64 | troppo larga |
| 204 | CHARISfive | ds | 12,412 | 11,043 | 1,114 | 2,355 | 3,596 | 32 | 4.00 | 64 | troppo larga |
| 205 | CHARISfive | run | 9,812 | 8,703 | 854 | 1,835 | 2,816 | 32 | 4.00 | 64 | troppo larga |
| 206 | CHARISten | ds | 25,130 | 22,490 | 2,386 | 4,899 | 7,412 | 36 | 3.56 | 72 | troppo larga |
| 207 | CHARISten | run | 3,165 | 2,721 | 189 | 506 | 822 | 39 | 3.28 | 78 | troppo larga |
| 208 | CHARISten | ds | 5,130 | 4,490 | 386 | 899 | 1,412 | 37 | 3.46 | 74 | troppo larga |
| 209 | Fantasia | run | 19,000 | 16,973 | 1,773 | 3,673 | 5,573 | 269 | 0.48 | 538 | MOLTO stretta |
| 210 | Italianpowerdemand | ds | 36,123 | 32,383 | 3,485 | 7,098 | 10,710 | 97 | 1.32 | 194 | un po' stretta |
| 211 | Italianpowerdemand | ds | 38,113 | 34,174 | 3,684 | 7,496 | 11,307 | 97 | 1.32 | 194 | un po' stretta |
| 212 | Italianpowerdemand | run | 8,913 | 7,894 | 764 | 1,656 | 2,547 | 25 | 5.12 | 50 | troppo larga |
| 213 | STAFFIIIDatabase | run | 33,211 | 29,762 | 3,194 | 6,515 | 9,836 | 878 | 0.15 | 1756 | MOLTO stretta |
| 214 | STAFFIIIDatabase | ds | 34,211 | 30,662 | 3,294 | 6,715 | 10,136 | 888 | 0.14 | 1776 | MOLTO stretta |
| 215 | STAFFIIIDatabase | ds | 36,276 | 32,521 | 3,500 | 7,128 | 10,756 | 851 | 0.15 | 1702 | MOLTO stretta |
| 216 | STAFFIIIDatabase | ds | 37,216 | 33,367 | 3,594 | 7,316 | 11,038 | 883 | 0.14 | 1766 | MOLTO stretta |
| 217 | STAFFIIIDatabase | ds | 38,211 | 34,262 | 3,694 | 7,515 | 11,336 | 891 | 0.14 | 1782 | MOLTO stretta |
| 218 | STAFFIIIDatabase | ds | 41,117 | 36,878 | 3,984 | 8,096 | 12,208 | 871 | 0.15 | 1742 | MOLTO stretta |
| 219 | STAFFIIIDatabase | ds | 41,612 | 37,323 | 4,034 | 8,195 | 12,356 | 872 | 0.15 | 1744 | MOLTO stretta |
| 220 | STAFFIIIDatabase | ds | 43,217 | 38,768 | 4,194 | 8,516 | 12,838 | 867 | 0.15 | 1734 | MOLTO stretta |
| 221 | STAFFIIIDatabase | ds | 45,616 | 40,927 | 4,434 | 8,996 | 13,558 | 891 | 0.14 | 1782 | MOLTO stretta |
| 222 | mit14046longtermecg | run | 56,123 | 50,383 | 5,485 | 11,098 | 16,710 | 101 | 1.27 | 202 | un po' stretta |
| 223 | mit14046longtermecg | ds | 74,123 | 66,583 | 7,285 | 14,698 | 22,110 | 98 | 1.31 | 196 | un po' stretta |
| 224 | mit14046longtermecg | ds | 76,123 | 68,383 | 7,485 | 15,098 | 22,710 | 98 | 1.31 | 196 | un po' stretta |
| 225 | mit14046longtermecg | ds | 81,214 | 72,965 | 7,994 | 16,116 | 24,237 | 105 | 1.22 | 210 | un po' stretta |
| 226 | mit14046longtermecg | ds | 96,123 | 86,383 | 9,485 | 19,098 | 28,710 | 105 | 1.22 | 210 | un po' stretta |
| 227 | mit14134longtermecg | ds | 11,231 | 9,980 | 996 | 2,119 | 3,242 | 230 | 0.56 | 460 | stretta |
| 228 | mit14134longtermecg | ds | 11,361 | 10,097 | 1,009 | 2,145 | 3,281 | 141 | 0.91 | 282 | stretta |
| 229 | mit14134longtermecg | ds | 16,363 | 14,599 | 1,509 | 3,146 | 4,782 | 127 | 1.01 | 254 | un po' stretta |
| 230 | mit14134longtermecg | ds | 19,363 | 17,299 | 1,809 | 3,746 | 5,682 | 134 | 0.96 | 268 | stretta |
| 231 | mit14134longtermecg | run | 8,763 | 7,759 | 749 | 1,626 | 2,502 | 123 | 1.04 | 246 | un po' stretta |
| 232 | mit14134longtermecg | ds | 8,763 | 7,759 | 749 | 1,626 | 2,502 | 123 | 1.04 | 246 | un po' stretta |
| 233 | mit14157longtermecg | run | 18,913 | 16,894 | 1,764 | 3,656 | 5,547 | 109 | 1.17 | 218 | un po' stretta |
| 234 | mit14157longtermecg | ds | 18,913 | 16,894 | 1,764 | 3,656 | 5,547 | 119 | 1.08 | 238 | un po' stretta |
| 235 | mit14157longtermecg | ds | 18,913 | 16,894 | 1,764 | 3,656 | 5,547 | 116 | 1.10 | 232 | un po' stretta |
| 236 | mit14157longtermecg | ds | 19,313 | 17,254 | 1,804 | 3,736 | 5,667 | 114 | 1.12 | 228 | un po' stretta |
| 237 | mit14157longtermecg | ds | 19,313 | 17,254 | 1,804 | 3,736 | 5,667 | 105 | 1.22 | 210 | un po' stretta |
| 238 | mit14157longtermecg | ds | 21,311 | 19,052 | 2,004 | 4,135 | 6,266 | 111 | 1.15 | 222 | un po' stretta |
| 239 | taichidbS0715Master | ds | 190,037 | 170,906 | 18,876 | 37,880 | 56,884 | 1496 | 0.09 | 2992 | MOLTO stretta |
| 240 | taichidbS0715Master | ds | 240,030 | 215,900 | 23,876 | 47,879 | 71,882 | 1514 | 0.08 | 3028 | MOLTO stretta |
| 241 | taichidbS0715Master | ds | 250,000 | 224,873 | 24,873 | 49,873 | 74,873 | 1496 | 0.09 | 2992 | MOLTO stretta |
| 242 | tilt12744mtable | ds | 100,000 | 89,873 | 9,873 | 19,873 | 29,873 | 223 | 0.57 | 446 | stretta |
| 243 | tilt12744mtable | ds | 100,000 | 89,873 | 9,873 | 19,873 | 29,873 | 223 | 0.57 | 446 | stretta |
| 244 | tilt12754table | ds | 100,013 | 89,884 | 9,874 | 19,876 | 29,877 | 162 | 0.79 | 324 | stretta |
| 245 | tilt12754table | ds | 100,211 | 90,062 | 9,894 | 19,915 | 29,936 | 166 | 0.77 | 332 | stretta |
| 246 | tilt12755mtable | ds | 100,211 | 90,062 | 9,894 | 19,915 | 29,936 | 221 | 0.58 | 442 | stretta |
| 247 | tilt12755mtable | run | 50,211 | 45,062 | 4,894 | 9,915 | 14,936 | 221 | 0.58 | 442 | stretta |
| 248 | weallwalk | NO | 2,000 | 1,673 | 73 | 273 | 473 | 27 | 4.74 | 54 | troppo larga |
| 249 | weallwalk | run | 2,753 | 2,350 | 148 | 424 | 699 | 26 | 4.92 | 52 | troppo larga |
| 250 | weallwalk | ds | 2,951 | 2,528 | 168 | 463 | 758 | 25 | 5.12 | 50 | troppo larga |
