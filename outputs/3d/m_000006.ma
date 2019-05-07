createNode transform -s -n "root_m_000006";
	rename -uid "e22253c0-5a68-49a7-9709-3c92c2c55478";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000006";
	rename -uid "63137b92-ee1a-4132-8e38-cefafd9c886d";
	setAttr ".t" -type "double3" -4.260289669036865 -12.501184642314911 0.05482230335474014 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "a531b435-f71e-4be0-9c2b-b8374f0dcf97";
	setAttr ".t" -type "double3" -8.66338312625885 -0.8779987692832947 -1.3516495004296303 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "218c3f96-7ef4-4452-8d0f-86b1bdadf6f7";
	setAttr ".t" -type "double3" -6.640322506427765 43.92622113227844 -4.828726127743721 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "d5ff225c-e122-4887-969c-47062de1a976";
	setAttr ".t" -type "double3" -6.6291555762290955 39.86608684062958 -4.059794172644615 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "3e4df7fe-d354-4d9f-8f61-46ed6344d0a6";
	setAttr ".t" -type "double3" 8.551419898867607 1.106008142232895 1.5303809195756912 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "66a0904e-db49-46c0-989a-acaad1b8311f";
	setAttr ".t" -type "double3" 2.1146733313798904 42.227139323949814 -4.470441676676273 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "1779aada-da5b-45ae-b827-f7b32abf74ac";
	setAttr ".t" -type "double3" 1.1775955557823181 40.11639952659607 -3.5897579044103622 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "0a2ef6d4-591f-4ea8-b3fe-fdfd134a14eb";
	setAttr ".t" -type "double3" 1.2551501393318176 -25.73893517255783 0.989221129566431 ;
createNode joint -n "neck" -p "spine";
	rename -uid "f4972876-4f17-4d08-86f5-3395a563b415";
	setAttr ".t" -type "double3" 1.3316642493009567 -26.317492127418518 3.2840964384377003 ;
createNode joint -n "head" -p "neck";
	rename -uid "d86047fb-61e7-4eb5-af9c-675c0866b7c7";
	setAttr ".t" -type "double3" 0.7951581850647926 -12.76782751083374 2.7603410184383392 ;
createNode joint -n "head_top" -p "head";
	rename -uid "b21c9cd3-03f3-453d-b7ef-e4cf1ff3a8d5";
	setAttr ".t" -type "double3" 1.0555524379014969 -12.72619366645813 12.63817846775055 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "d1108c73-e614-4928-b5b6-7844fd8397b2";
	setAttr ".t" -type "double3" -17.090165242552757 2.3865103721618652 -4.6634008176624775 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "3657578b-b108-4dc6-b77d-4a693f3362c8";
	setAttr ".t" -type "double3" -25.33906400203705 1.2104213237762451 -6.3229660503566265 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "7765d44c-cf49-4945-9fd1-d74616e12f01";
	setAttr ".t" -type "double3" -27.854710817337036 -0.7360517978668213 -6.510292738676071 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "17541d0c-48dd-4b8d-a406-48afb928d20e";
	setAttr ".t" -type "double3" 16.4278332144022 4.786384105682373 -0.020581483840942383 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "9c364f2f-27b0-469b-b7bf-f547dd5cc9f6";
	setAttr ".t" -type "double3" 24.503472447395325 4.3205201625823975 4.403425008058548 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "1ae03442-df39-44aa-b372-4e820350a2f4";
	setAttr ".t" -type "double3" 23.57785701751709 4.13927435874939 1.9893720746040344 ;
