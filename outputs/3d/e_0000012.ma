createNode transform -s -n "root_e_0000012";
	rename -uid "d91c4bd3-d5c5-4c57-b356-d94082847b47";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_e_0000012";
	rename -uid "235b267c-eb9f-4715-a139-61215aa1a0a7";
	setAttr ".t" -type "double3" 1.111423783004284 -2.665669284760952 0.7598081603646278 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "29f47807-476e-4c1c-aa74-2f2e994fbe58";
	setAttr ".t" -type "double3" 10.421408899128437 5.469761043787003 -4.381959326565266 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "979f6230-eb29-49d4-b311-65e663d8087a";
	setAttr ".t" -type "double3" -38.777194917201996 40.8021179959178 35.380954295396805 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "1d0baa9c-0b1d-4df1-9ddc-9d48d9c49580";
	setAttr ".t" -type "double3" -30.71863353252411 31.005793809890747 17.20675528049469 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "3881745d-a693-43cd-8770-faed7b9aca22";
	setAttr ".t" -type "double3" -9.591746889054775 -3.1652258709073067 4.274054430425167 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "f8612ef3-3679-49ac-b711-7fc23a47a4af";
	setAttr ".t" -type "double3" -21.635928004980087 49.252354353666306 32.15034604072571 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "bb586947-eb0b-4fc4-b6a5-ff7cf9b29eee";
	setAttr ".t" -type "double3" -15.27467668056488 40.6779408454895 14.716076850891113 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "af8a8139-0a8a-4c5b-a23e-f6d5a07e52cd";
	setAttr ".t" -type "double3" 28.16732544451952 -27.60069314390421 -8.26553050428629 ;
createNode joint -n "neck" -p "spine";
	rename -uid "05c0c103-5cb4-48d1-a765-03a3cf87b368";
	setAttr ".t" -type "double3" 34.802764654159546 -37.70636320114136 12.529339268803596 ;
createNode joint -n "head" -p "neck";
	rename -uid "b1b1b62e-5b92-47ed-8b5b-cea02f9e4d6e";
	setAttr ".t" -type "double3" 16.67216420173645 -8.267641067504883 11.93695031106472 ;
createNode joint -n "head_top" -p "head";
	rename -uid "9808bf84-b4c2-452b-92b1-37ee5ed0ce8a";
	setAttr ".t" -type "double3" 11.276662349700928 -5.777931213378906 19.019317626953125 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "247cb8c0-ea40-4c92-82c3-b40b7223c998";
	setAttr ".t" -type "double3" 1.1384427547454834 35.72050929069519 -11.484842374920845 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "2086d317-2ad1-42b6-b043-789e5b0729b9";
	setAttr ".t" -type "double3" -29.795542359352112 1.5815258026123047 -12.499631941318512 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "66440d04-fab2-499a-804a-c8612855fb60";
	setAttr ".t" -type "double3" -11.60794049501419 -5.352628231048584 55.04733920097351 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "ea209ef2-9966-4dc3-a294-3ed06f2c57bb";
	setAttr ".t" -type "double3" -14.659151434898376 -3.311210870742798 6.422930583357811 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "a459edbb-ca3c-4889-9016-b085631cedce";
	setAttr ".t" -type "double3" -24.614322185516357 17.928725481033325 25.270122289657593 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "60092361-b2e2-49e6-b336-673a2a6275c9";
	setAttr ".t" -type "double3" -0.7011741399765015 8.406174182891846 18.95262598991394 ;
