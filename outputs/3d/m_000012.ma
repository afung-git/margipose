createNode transform -s -n "root_m_000012";
	rename -uid "46fb9b54-d100-44d2-beb6-1875490857da";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000012";
	rename -uid "1fa7ba04-f7a6-466c-819b-ccea1a08a141";
	setAttr ".t" -type "double3" -3.091268613934517 4.5646995306015015 -0.0529823824763298 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "d4a81f86-ff66-4e45-a174-c25ba0fe6c2f";
	setAttr ".t" -type "double3" -8.871927484869957 2.729921042919159 -2.8342856094241142 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "ddb6b707-3dc6-4591-a2fc-f8b66a8e6843";
	setAttr ".t" -type "double3" 3.97522896528244 37.5757172703743 2.063293196260929 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "27ab818e-599a-48aa-b45e-bc3254378b16";
	setAttr ".t" -type "double3" -1.435687392950058 32.74970352649689 13.371248729526997 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "4d192751-16f1-48fc-89bc-f105e804b72c";
	setAttr ".t" -type "double3" 8.446814492344856 -2.392422780394554 2.425411343574524 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "ad9c4cb8-7856-484e-a0b2-c253f30da189";
	setAttr ".t" -type "double3" 17.830054461956024 29.024359211325645 -40.57131130248308 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "b0557c5d-dcc0-4a7d-9729-f051b7c17d94";
	setAttr ".t" -type "double3" 15.012119710445404 36.077624559402466 -5.815345048904419 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "cb889c17-4edc-4829-b93a-0a599709b32f";
	setAttr ".t" -type "double3" -8.359986916184425 -21.20928466320038 -3.8307437673211098 ;
createNode joint -n "neck" -p "spine";
	rename -uid "b8810bd5-50b6-4873-b082-19d94f1d192e";
	setAttr ".t" -type "double3" -8.759421855211258 -18.474090099334717 -12.92746290564537 ;
createNode joint -n "head" -p "neck";
	rename -uid "22997cf7-f148-4bd9-a58e-57ab32dd272a";
	setAttr ".t" -type "double3" -4.124324023723602 -12.406158447265625 -6.490936875343323 ;
createNode joint -n "head_top" -p "head";
	rename -uid "4e900e0f-5a9b-4d9d-bed0-af014da3ef86";
	setAttr ".t" -type "double3" -4.127253592014313 -13.003075122833252 -7.2709619998931885 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "096a5e85-ebf3-4322-a8f8-e60baa3eba32";
	setAttr ".t" -type "double3" -12.815943360328674 12.640134990215302 -0.7683515548706055 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "0320b4a7-e834-404e-991c-18ad2fe0229d";
	setAttr ".t" -type "double3" -2.8690457344055176 26.52839496731758 1.8912628293037415 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "5c7b8944-61b7-4199-b4e7-b86ed09d1c07";
	setAttr ".t" -type "double3" -5.741119384765625 22.360161691904068 -6.718756258487701 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "38839bd3-18bf-433b-9174-7a6d6a40270e";
	setAttr ".t" -type "double3" 12.990496307611465 -11.590707302093506 5.19133061170578 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "cfb21dc5-7558-4b74-8737-fe2be6523879";
	setAttr ".t" -type "double3" 19.07597780227661 -14.599668979644775 2.7326874434947968 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "97f28723-e7e9-45e1-a0ac-b53a236e6272";
	setAttr ".t" -type "double3" 12.022707611322403 -16.936737298965454 -3.589925915002823 ;
