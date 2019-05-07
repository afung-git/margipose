createNode transform -s -n "root_m_000022";
	rename -uid "582c7b7d-b0cb-4c88-a9f7-91bcffce1380";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000022";
	rename -uid "ef02f323-dd4f-4ca6-a8b5-1ecaa2809599";
	setAttr ".t" -type "double3" -8.113174885511398 -3.1384609639644623 -0.06471080705523491 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "93cadc2d-c125-4f67-a4da-75d6dd5bff62";
	setAttr ".t" -type "double3" -7.466230541467667 0.9973134845495224 -4.15326589718461 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "cf4cc3d0-6bc0-4f40-baaa-1dac005a78bb";
	setAttr ".t" -type "double3" 2.68523246049881 33.382799848914146 14.540008082985878 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "ff4efe7a-457e-481b-bf4a-353b629563bd";
	setAttr ".t" -type "double3" 6.900283321738243 33.25870931148529 14.624208211898804 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "bfad50bd-2cbe-48c6-a441-fa0842bc5932";
	setAttr ".t" -type "double3" 7.355455495417118 -0.6950553506612778 2.71617965772748 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "e739772e-c227-457a-824b-e0f5cd9ad8f1";
	setAttr ".t" -type "double3" 6.29825945943594 32.54818208515644 7.6179539784789085 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "4f7f5310-13d0-4369-9a47-417492bd7359";
	setAttr ".t" -type "double3" 4.469127207994461 32.18665421009064 11.084084212779999 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "e232af00-e5e3-4be6-811c-272c7a1b0883";
	setAttr ".t" -type "double3" -5.582382529973984 -17.846450954675674 -13.025813270360231 ;
createNode joint -n "neck" -p "spine";
	rename -uid "e7499289-fc9a-47e4-859b-aa7cf9d99f58";
	setAttr ".t" -type "double3" -5.992519855499268 -14.090362191200256 -21.41401469707489 ;
createNode joint -n "head" -p "neck";
	rename -uid "4ece7ec2-52ee-4626-b346-2eb119be8cdc";
	setAttr ".t" -type "double3" -4.523774981498718 -12.42567002773285 -7.339385151863098 ;
createNode joint -n "head_top" -p "head";
	rename -uid "b23d7c41-3d86-43d1-9434-5924c906563d";
	setAttr ".t" -type "double3" -4.603083431720734 -12.829065322875977 -9.696629643440247 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "1093c662-140f-4367-b78e-a5339ab44b3b";
	setAttr ".t" -type "double3" -12.729717791080475 11.037558317184448 -1.3762116432189941 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "ecc0877d-6c39-461c-9c90-39ac444b6312";
	setAttr ".t" -type "double3" -4.309892654418945 25.24687424302101 -3.071242570877075 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "8c692da5-0737-44d0-b89a-1a010a3f4b6f";
	setAttr ".t" -type "double3" -4.413628578186035 24.838297814130783 -5.026093125343323 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "c8ea05cb-bfa4-4026-90b4-45a7d6e6c86b";
	setAttr ".t" -type "double3" 12.092305719852448 -11.828875541687012 5.4962158203125 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "964a7a46-53ec-44ce-be12-d804ed0607a2";
	setAttr ".t" -type "double3" 16.904309391975403 -9.951457381248474 3.1985610723495483 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "242a90e4-d39c-4b3e-9af3-ce2126f93771";
	setAttr ".t" -type "double3" 13.644196093082428 -19.177448749542236 -7.223302125930786 ;
