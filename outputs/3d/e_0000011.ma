createNode transform -s -n "root_e_0000011";
	rename -uid "5c203d1e-fbae-4d83-9eb9-8fc746290386";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_e_0000011";
	rename -uid "be3a9cd7-1309-4a7f-9c14-36099c35649c";
	setAttr ".t" -type "double3" -5.510298162698746 -2.676270343363285 -0.13177776709198952 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "22046a9b-29c0-47a2-b032-5081b656b5b7";
	setAttr ".t" -type "double3" -10.447648912668228 1.4171682298183441 -5.752904620021582 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "7a382175-c899-46cd-90d5-0e326e33ba4d";
	setAttr ".t" -type "double3" 10.900949314236641 37.14658487588167 17.64748841524124 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "a3f0f5a8-9827-49a7-8d5b-aa4d60ac74fb";
	setAttr ".t" -type "double3" 7.975944504141808 35.04713475704193 19.56559792160988 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "8f081eb3-c690-43a6-bdfe-dda6123a79ed";
	setAttr ".t" -type "double3" 9.569982439279556 -1.8666697666049004 5.4079812951385975 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "286f28b3-6243-468e-878f-2e237af095e7";
	setAttr ".t" -type "double3" 1.596042513847351 40.503304451704025 26.787107810378075 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "e871b6dc-adea-4a69-b39a-c9d950037431";
	setAttr ".t" -type "double3" -3.5276858136057854 19.14982795715332 24.486225843429565 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "4dbae712-bade-4610-b115-2541b219a35b";
	setAttr ".t" -type "double3" -5.192997306585312 -28.607012890279293 -11.421897169202566 ;
createNode joint -n "neck" -p "spine";
	rename -uid "5aa56040-7012-4c28-b569-cc8edfc5d47f";
	setAttr ".t" -type "double3" -4.145435988903046 -34.92538332939148 -15.84567129611969 ;
createNode joint -n "head" -p "neck";
	rename -uid "6d6294c8-4ef7-4c83-89d9-cbe4b8ce7b02";
	setAttr ".t" -type "double3" -0.7077977061271667 -14.338719844818115 -6.459769606590271 ;
createNode joint -n "head_top" -p "head";
	rename -uid "e51d917e-6c21-4baa-8c55-496a49b42aa3";
	setAttr ".t" -type "double3" -0.551179051399231 -12.640446424484253 -6.629389524459839 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "7c35b0db-aee9-4d66-9db3-903d3153e2bd";
	setAttr ".t" -type "double3" -22.69348055124283 12.895405292510986 0.5498558282852173 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "0fafc083-2699-405f-a609-b249b11661c1";
	setAttr ".t" -type "double3" -2.2257328033447266 28.880664706230164 13.655997812747955 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "6c4752c1-e252-49a0-9d65-74f0d26868bc";
	setAttr ".t" -type "double3" 7.749032974243164 6.450861692428589 -14.897222816944122 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "05e20096-de40-43f9-92b1-d9971d3a4c7f";
	setAttr ".t" -type "double3" 24.47296231985092 5.672407150268555 2.6515886187553406 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "313184f1-067e-4213-9873-64c26bb44383";
	setAttr ".t" -type "double3" 12.75700032711029 19.368550181388855 3.166690468788147 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "66514523-227a-4a45-8cff-2c7b9d46671e";
	setAttr ".t" -type "double3" -17.70200878381729 10.36858856678009 -22.013376653194427 ;
