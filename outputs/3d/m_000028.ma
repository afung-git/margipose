createNode transform -s -n "root_m_000028";
	rename -uid "80597ff0-761e-4a1a-9a51-a22f722a2243";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000028";
	rename -uid "e0e23dac-4e81-45c7-b782-e31452743557";
	setAttr ".t" -type "double3" -1.105421595275402 -23.61202985048294 -0.23165661841630936 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "c7f3f987-637d-489a-9d22-5494c32b2610";
	setAttr ".t" -type "double3" 4.525736533105373 -0.21621286869049072 -0.5176573060452938 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "29f76ee8-bc34-48c7-a200-8651e7d757f0";
	setAttr ".t" -type "double3" -1.5061615034937859 25.03932435065508 12.950697634369135 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "e91ab583-19a1-4fb4-bde6-d9a51ca7e3da";
	setAttr ".t" -type "double3" -2.5936750695109367 28.135151974856853 -11.494499444961548 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "7e21a85f-9ec1-4a96-91fb-c34f49bb770d";
	setAttr ".t" -type "double3" -4.83716893941164 0.07403194904327393 -0.48234108835458755 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "9cd3923b-3367-481a-8aa2-1e905a72d00a";
	setAttr ".t" -type "double3" 3.227916918694973 25.05535762757063 -4.276244714856148 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "5b69038b-018f-4fcc-a180-78cc7861a0ff";
	setAttr ".t" -type "double3" 2.5379711762070656 25.70429015904665 -11.829055845737457 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "547d93f2-6ec5-482f-980f-14e06e0d9f52";
	setAttr ".t" -type "double3" -0.9382665157318115 -17.64778047800064 3.92488706856966 ;
createNode joint -n "neck" -p "spine";
	rename -uid "2396105c-0f4b-4bb4-8da2-db6d777a0d8b";
	setAttr ".t" -type "double3" 0.19259396940469742 -20.451632142066956 7.313603907823563 ;
createNode joint -n "head" -p "neck";
	rename -uid "3c9a3e56-c596-49a7-b732-2b6c823a1517";
	setAttr ".t" -type "double3" 0.31621307134628296 -10.806608200073242 2.865590900182724 ;
createNode joint -n "head_top" -p "head";
	rename -uid "2593f9ce-6a81-45ba-b803-5da4c0e93f4b";
	setAttr ".t" -type "double3" 0.1993754878640175 -12.064766883850098 3.576451539993286 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "a757af2b-06ba-4a5f-b945-c9850b924628";
	setAttr ".t" -type "double3" 8.967948332428932 2.377241849899292 -0.9874023497104645 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "ca23e64d-d0b0-4979-94d4-f06e6ec92261";
	setAttr ".t" -type "double3" 8.96102637052536 13.095691800117493 22.732307016849518 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "54e569f7-df72-4047-a76a-6e8fd86817be";
	setAttr ".t" -type "double3" -4.552410542964935 0.9866416454315186 21.62981629371643 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "52d40b6f-be13-435a-81f9-619b4cba316e";
	setAttr ".t" -type "double3" -9.80309210717678 4.345595836639404 3.6879248917102814 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "c13a9e5c-552b-4a21-92c4-109208a9279d";
	setAttr ".t" -type "double3" 0.05923360586166382 14.46845531463623 23.77166748046875 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "8796654d-c215-4b15-bff2-72fe8bba61ca";
	setAttr ".t" -type "double3" 9.168951958417892 1.149839162826538 25.33104121685028 ;
