createNode transform -s -n "root_WP_20190505_11_28_02_Raw";
	rename -uid "69d7f015-27f6-4a13-b62a-d01e3282b9fb";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_WP_20190505_11_28_02_Raw";
	rename -uid "da6b96a8-0c20-4867-9e26-ca88bf29081c";
	setAttr ".t" -type "double3" 1.3718778267502785 -6.831977516412735 0.60140211135149 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "45a0edd6-43ba-4570-9a0a-f2067bda5f83";
	setAttr ".t" -type "double3" 4.015586338937283 3.8086656481027603 3.503573499619961 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "8fd01a2d-2656-4cde-94cd-09de5779a207";
	setAttr ".t" -type "double3" 3.64358127117157 -25.753798708319664 -82.7283538877964 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "21f1bf8f-03c7-4720-b08d-db92a65e8127";
	setAttr ".t" -type "double3" -26.981230080127716 -49.36978816986084 119.03984546661377 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "d390e379-0496-4975-af66-9443b3699101";
	setAttr ".t" -type "double3" -0.824580155313015 -2.5379516184329987 -2.6383941993117332 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "f78562c0-19fd-41e3-9e92-65ec2d059012";
	setAttr ".t" -type "double3" 2.782197669148445 -20.010222494602203 3.1536098569631577 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "0b73076b-7630-4da0-a417-0a6382048d76";
	setAttr ".t" -type "double3" 51.75301656126976 -4.188892245292664 37.79347315430641 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "8f98f2ed-95b9-4bed-a9dd-b282edd062f7";
	setAttr ".t" -type "double3" -24.954099394381046 8.801992796361446 4.714866541326046 ;
createNode joint -n "neck" -p "spine";
	rename -uid "f1bdc9f8-884f-4364-95dc-c166c5224815";
	setAttr ".t" -type "double3" -28.022246062755585 9.404651261866093 4.6211376786231995 ;
createNode joint -n "head" -p "neck";
	rename -uid "00ae35e9-79c6-4371-8a99-1318e033a52a";
	setAttr ".t" -type "double3" -11.649876832962036 0.718146562576294 2.281578630208969 ;
createNode joint -n "head_top" -p "head";
	rename -uid "b263c05a-0f6c-4b48-9750-f47f76378e1e";
	setAttr ".t" -type "double3" -15.865945816040039 4.591403156518936 -43.40157210826874 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "36285de9-ae1d-410f-9df8-37c1db60fa5b";
	setAttr ".t" -type "double3" 8.676573634147644 10.850020498037338 2.5109075009822845 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "ed827696-b66c-4ede-8744-95754bf465ee";
	setAttr ".t" -type "double3" 11.745968461036682 18.589697778224945 -12.212525308132172 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "4bb0728b-70b6-4887-a30b-45e873de320f";
	setAttr ".t" -type "double3" 15.011923015117645 21.447739005088806 -19.30752843618393 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "b1390dd5-f824-4ee4-a058-e4c6889565b9";
	setAttr ".t" -type "double3" -8.4042489528656 -9.72956009209156 -1.164538413286209 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "c19dbe0d-a5a5-49ac-84c2-9dd9b4e3ea8f";
	setAttr ".t" -type "double3" -4.385823011398315 -6.395531445741653 3.52671816945076 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "b30c08f3-cd08-4efc-ab5f-b11f7c4adc96";
	setAttr ".t" -type "double3" -18.13783049583435 -9.196064993739128 38.24755474925041 ;