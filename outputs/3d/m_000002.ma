createNode transform -s -n "root_m_000002";
	rename -uid "f560fb49-66bb-4c95-8e2f-2aafb97ae8da";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000002";
	rename -uid "6c05a63d-9f73-4289-8b5b-66be9a2cd088";
	setAttr ".t" -type "double3" -4.1052162647247314 -13.061292469501495 0.12802490964531898 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "f2fcfb39-4b51-4676-9762-8acca360fde5";
	setAttr ".t" -type "double3" -8.543801307678223 -0.8164837956428528 -1.8743154592812061 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "5bff7210-660a-4475-9701-3fc70d81795d";
	setAttr ".t" -type "double3" -6.895598769187927 45.041438937187195 -2.388599142432213 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "15d9662c-f992-4b72-b70c-9df63a7cf880";
	setAttr ".t" -type "double3" -6.7009806632995605 39.12419676780701 -2.0553715527057648 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "aaa4ed50-de82-48df-a084-2002b88e56aa";
	setAttr ".t" -type "double3" 8.351396024227142 1.2292370200157166 2.01009726151824 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "7e76b141-d847-40d5-b99a-5909d4ef8b27";
	setAttr ".t" -type "double3" 2.2495992481708527 43.08183789253235 -3.3473487943410873 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "57c85fcb-add9-42cf-8fcd-7988cc2d12f7";
	setAttr ".t" -type "double3" 1.1026635766029358 39.70203101634979 -1.1795185506343842 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "dea7e700-eba5-4c97-a419-c54f4980dea1";
	setAttr ".t" -type "double3" 1.1490058153867722 -25.70304423570633 -0.03343923017382622 ;
createNode joint -n "neck" -p "spine";
	rename -uid "e0340bf5-cc8f-4659-8c6b-30ca66f27a06";
	setAttr ".t" -type "double3" 2.45477557182312 -33.27659070491791 -0.1377093605697155 ;
createNode joint -n "head" -p "neck";
	rename -uid "35794cbd-fedd-43a1-b2a1-41885aeb7549";
	setAttr ".t" -type "double3" 1.4316966757178307 -13.257050514221191 1.2441925704479218 ;
createNode joint -n "head_top" -p "head";
	rename -uid "20577d0c-3af3-432a-a7a5-c942b9f6adf5";
	setAttr ".t" -type "double3" 1.5254681929945946 -9.171086549758911 8.838428277522326 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "e85eb4af-b3ac-4b0e-89b7-728d143f4f05";
	setAttr ".t" -type "double3" -18.899090215563774 9.64118242263794 -5.180090572685003 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "8b11a18b-c243-431c-96e5-05b62653163e";
	setAttr ".t" -type "double3" -25.354093313217163 1.3258755207061768 -4.602788761258125 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "ba44c1c4-7787-433b-bc38-f7166c43f324";
	setAttr ".t" -type "double3" -27.434641122817993 -0.6664037704467773 -6.1310261487960815 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "ecc3630a-ac40-4ed3-8371-5acda12dcef7";
	setAttr ".t" -type "double3" 15.346366539597511 11.519384384155273 1.7971950583159924 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "f18f5c87-9e46-4d50-a028-8e39fee1eae0";
	setAttr ".t" -type "double3" 24.027757346630096 4.948478937149048 7.223060168325901 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "58a3782d-712a-4485-883d-520c3c854d1f";
	setAttr ".t" -type "double3" 23.444640636444092 4.0399134159088135 2.793194353580475 ;
