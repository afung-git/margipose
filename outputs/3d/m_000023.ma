createNode transform -s -n "root_m_000023";
	rename -uid "b0680611-0480-45f1-9a3f-2cdfd09a9318";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000023";
	rename -uid "b736a02d-6a05-45eb-82d9-9e83c759687b";
	setAttr ".t" -type "double3" -4.283209890127182 4.270347952842712 -0.02882331609725952 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "d3ea37b5-d943-42f6-97a5-e70e8ff63556";
	setAttr ".t" -type "double3" -7.673627883195877 2.2261448204517365 -6.462588906288147 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "1923b193-7047-4bf9-9d0b-7be68dfa71af";
	setAttr ".t" -type "double3" 4.133283346891403 37.836237996816635 7.214460242539644 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "47599577-3687-451f-a628-1024c245a5b5";
	setAttr ".t" -type "double3" -1.4265641570091248 33.265846967697144 10.424322914332151 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "6de74c5c-c0a8-4ad5-bcfa-41d139a2a97b";
	setAttr ".t" -type "double3" 7.79992938041687 -2.5386258959770203 6.01545013487339 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "5d176e19-b235-447a-834b-fb8863a05438";
	setAttr ".t" -type "double3" 18.53635385632515 33.324335515499115 -8.808713220059872 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "96e2bd20-a420-47ff-b212-54e4d01f81ca";
	setAttr ".t" -type "double3" 16.61372035741806 33.47046077251434 1.60555737093091 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "775eb966-6675-46bf-8c8e-75f69dc0ff94";
	setAttr ".t" -type "double3" -7.656632363796234 -22.827018797397614 -1.4255479909479618 ;
createNode joint -n "neck" -p "spine";
	rename -uid "d70dfe35-b64a-4399-b4d9-4d2caacfdf99";
	setAttr ".t" -type "double3" -6.021375209093094 -21.606425940990448 -1.0718557052314281 ;
createNode joint -n "head" -p "neck";
	rename -uid "ed2bea93-f992-4201-a08c-af39d1ea956f";
	setAttr ".t" -type "double3" -4.14658784866333 -9.364187717437744 0.47678034752607346 ;
createNode joint -n "head_top" -p "head";
	rename -uid "7b64d47a-603e-4d79-8263-37664b183998";
	setAttr ".t" -type "double3" -8.23369175195694 -11.400255560874939 5.368162877857685 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "f3e6ac2a-9435-491a-be24-edfa8d768563";
	setAttr ".t" -type "double3" -12.225662171840668 12.35230565071106 -9.833507612347603 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "c83bf0e1-9538-40cc-aaf6-7517eeea63ee";
	setAttr ".t" -type "double3" -5.880075693130493 27.20159851014614 -10.047309845685959 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "828325fd-2a55-4148-a4cf-97d9bd83dad1";
	setAttr ".t" -type "double3" -5.341529846191406 26.62614919245243 -10.12212485074997 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "bf74b099-9be3-47f9-ab66-3f9048516be2";
	setAttr ".t" -type "double3" 12.68075406551361 -7.645618915557861 7.195967435836792 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "eb7def75-c436-4a08-9652-de4bf5988f0c";
	setAttr ".t" -type "double3" 15.852490812540054 -10.11333167552948 -2.381514199078083 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "d9741b0b-cba9-4117-a8ba-032ee77b6cd8";
	setAttr ".t" -type "double3" 12.75373324751854 -19.096213579177856 -0.5879638716578484 ;