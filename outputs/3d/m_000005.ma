createNode transform -s -n "root_m_000005";
	rename -uid "a73151af-ab90-4bd3-8ea9-8284d7b1ee3d";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000005";
	rename -uid "393e4e4b-658c-4b11-b233-896f00a05eb1";
	setAttr ".t" -type "double3" -4.278190806508064 -12.546244263648987 0.08730581030249596 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "89e33bf4-2666-4c4a-ada9-b7fd09198bf8";
	setAttr ".t" -type "double3" -8.468412980437279 -0.830136239528656 -1.727792900055647 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "3d97fd5b-8d0c-4891-96fa-40b1943e9a96";
	setAttr ".t" -type "double3" -6.83235228061676 42.92233735322952 -0.09136833250522614 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "405e57c3-f19e-43f9-9de6-c7ea0459db7c";
	setAttr ".t" -type "double3" -6.483443081378937 40.32418131828308 -5.180030316114426 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "0a660ab6-5954-4a9f-9b18-49bb8c91710b";
	setAttr ".t" -type "double3" 8.330698311328888 1.096481829881668 2.4083097465336323 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "fd849771-0cbc-4821-a01b-270e038ad435";
	setAttr ".t" -type "double3" 1.8871303647756577 41.03010520339012 -0.36257319152355194 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "cf8879d4-5c9c-4310-9221-894200797edb";
	setAttr ".t" -type "double3" 1.345747709274292 39.82861042022705 -0.8167184889316559 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "b72f7a36-a4db-44cc-afb8-63c40763c3a1";
	setAttr ".t" -type "double3" 1.6394088044762611 -25.760260224342346 -0.8956572972238064 ;
createNode joint -n "neck" -p "spine";
	rename -uid "be5ea837-0a41-4f81-9991-9b84f6b9af23";
	setAttr ".t" -type "double3" 2.322573959827423 -33.672916889190674 -1.2848064303398132 ;
createNode joint -n "head" -p "neck";
	rename -uid "e70bb376-11b5-43da-9baa-5225214a3cdc";
	setAttr ".t" -type "double3" 1.344488002359867 -13.249850273132324 1.582676637917757 ;
createNode joint -n "head_top" -p "head";
	rename -uid "cf7eca82-ffa5-4976-951a-b20c736abb7b";
	setAttr ".t" -type "double3" 1.5003345906734467 -9.153246879577637 10.312296729534864 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "d9ce7fe6-4ef5-4b33-be03-4b603cf3ff3f";
	setAttr ".t" -type "double3" -18.939112685620785 9.126335382461548 -5.259393900632858 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "13350176-0d17-445e-a238-95f2c93d3a37";
	setAttr ".t" -type "double3" -25.814224779605865 2.76830792427063 -2.489311993122101 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "9dfb9010-741c-4e3c-bcae-8c2112160c29";
	setAttr ".t" -type "double3" -27.761459350585938 -1.6494214534759521 -12.891069054603577 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "700f3b21-3da5-407f-90a4-7bbc97abdadd";
	setAttr ".t" -type "double3" 15.918443538248539 11.462801694869995 3.2671142369508743 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "d936d5f8-7aad-4de4-b5a3-f9913bfdb5ae";
	setAttr ".t" -type "double3" 24.516601860523224 5.172985792160034 11.300880834460258 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "3e509b2f-525c-4071-b863-02eefcaa93e9";
	setAttr ".t" -type "double3" 23.595023155212402 3.5393238067626953 4.398847371339798 ;
