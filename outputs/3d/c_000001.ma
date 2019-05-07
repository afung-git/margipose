createNode transform -s -n "root_c_000001";
	rename -uid "517a2f7e-997b-4aad-9b1d-7e9a30650d51";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_c_000001";
	rename -uid "af607817-b3b1-404c-877e-2e2abaf782e3";
	setAttr ".t" -type "double3" -6.531890481710434 -15.731216967105865 -1.1872564442455769 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "2dfe25e8-e3c6-4cdb-bbe7-aef075665afa";
	setAttr ".t" -type "double3" 9.478466585278511 -1.5301331877708435 -1.7203892581164837 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "be20b99b-fec1-46b3-8d80-a724480f2a5e";
	setAttr ".t" -type "double3" 15.298658981919289 -29.680868983268738 -12.324923276901245 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "8ee6c9c2-99bf-4a98-a0c5-d4431f6f110b";
	setAttr ".t" -type "double3" 7.427451014518738 -12.887701392173767 6.351737678050995 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "cf65c181-b04b-4e7e-9d4f-686b4d297f43";
	setAttr ".t" -type "double3" -10.210082679986954 -0.5483746528625488 1.1658607050776482 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "a56bed03-3833-4a43-b088-3a609915b90a";
	setAttr ".t" -type "double3" 6.459292024374008 -19.01848167181015 24.376100953668356 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "9330b217-b83e-44a6-aa1c-20de97f6e456";
	setAttr ".t" -type "double3" 26.633910089731216 -16.9286847114563 0.18914341926574707 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "b1828e67-bf2f-4873-9f92-89ffafa06f77";
	setAttr ".t" -type "double3" 4.540218599140644 7.1180544793605804 1.8458444625139236 ;
createNode joint -n "neck" -p "spine";
	rename -uid "fdba086b-c4e2-4ff1-b6da-903068662bfb";
	setAttr ".t" -type "double3" 4.990638233721256 19.68500390648842 1.689284574240446 ;
createNode joint -n "head" -p "neck";
	rename -uid "cd718fb6-812c-4522-833e-6646951c0a14";
	setAttr ".t" -type "double3" -0.22141225636005402 18.546102941036224 -4.223527200520039 ;
createNode joint -n "head_top" -p "head";
	rename -uid "51677d13-62f3-4c72-8ae3-caabd6556567";
	setAttr ".t" -type "double3" 3.1835541129112244 -6.425128877162933 -5.3171806037425995 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "e9099c79-7bda-4ba3-9179-57c6f823b8c9";
	setAttr ".t" -type "double3" 21.30800448358059 -5.91643825173378 -8.767207153141499 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "94bf1d68-1153-4a40-a64c-3fa36951f0b8";
	setAttr ".t" -type "double3" 23.247143626213074 -14.855343848466873 -13.38806226849556 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "5243f677-aa11-49a1-976b-3baa0c58330b";
	setAttr ".t" -type "double3" -3.7589341402053833 -21.018433570861816 -11.663036048412323 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "fbf90b38-8338-49f2-9e4c-608956051f7d";
	setAttr ".t" -type "double3" -18.16788651049137 -8.687413111329079 2.508680336177349 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "531d9d11-2587-4181-b07d-a977c3183d84";
	setAttr ".t" -type "double3" -15.82387387752533 -15.358246490359306 -22.476499527692795 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "6af32741-1ea3-4346-a7e2-6adb699536f1";
	setAttr ".t" -type "double3" -12.587970495223999 -0.20653456449508667 -10.39610505104065 ;
