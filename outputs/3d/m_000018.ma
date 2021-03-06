createNode transform -s -n "root_m_000018";
	rename -uid "6839cd7b-d974-43c4-8ecf-284c58b7fdb1";
	setAttr ".r" -type "double3" 180 0 0 ;
createNode joint -n "pelvis" -p "root_m_000018";
	rename -uid "1a18b59f-43db-4d52-a978-ab911bb2a80c";
	setAttr ".t" -type "double3" -3.154739737510681 5.800009518861771 -0.15014102682471275 ;
createNode joint -n "r_hip" -p "pelvis";
	rename -uid "8363900e-d4ff-449a-a01a-2e723b3a096a";
	setAttr ".t" -type "double3" -8.505959808826447 2.369481325149536 0.18907207995653152 ;
createNode joint -n "r_knee" -p "r_hip";
	rename -uid "86fbeba0-e7ec-4b84-9517-c843dcccad66";
	setAttr ".t" -type "double3" 4.271124303340912 34.982382506132126 0.912229809910059 ;
createNode joint -n "r_ankle" -p "r_knee";
	rename -uid "814c8d5a-74ce-416d-8c1b-625f74a2cf80";
	setAttr ".t" -type "double3" -0.8601225912570953 34.9027156829834 29.5941274613142 ;
createNode joint -n "l_hip" -p "pelvis";
	rename -uid "60f9263b-d009-4c8a-9c24-720332d34ba2";
	setAttr ".t" -type "double3" 8.898485451936722 -1.7743319272994995 -0.11784248054027557 ;
createNode joint -n "l_knee" -p "l_hip";
	rename -uid "26289835-8512-42fe-b73c-e932b5c1a9cd";
	setAttr ".t" -type "double3" 16.540073603391647 26.939602941274643 -47.04661266878247 ;
createNode joint -n "l_ankle" -p "l_knee";
	rename -uid "33354cf4-5ea5-49a1-987b-ecbc4ed171f4";
	setAttr ".t" -type "double3" 15.552443265914917 36.57410740852356 -15.782195329666138 ;
createNode joint -n "spine" -p "pelvis";
	rename -uid "671e1437-c268-4048-94b5-ad1f04b0f994";
	setAttr ".t" -type "double3" -8.179108053445816 -21.75181433558464 -0.9247125126421452 ;
createNode joint -n "neck" -p "spine";
	rename -uid "bf46578d-4a00-4984-9952-2f6c8d17a672";
	setAttr ".t" -type "double3" -8.227633684873581 -19.400404393672943 -7.454191148281097 ;
createNode joint -n "head" -p "neck";
	rename -uid "f2730a03-3ca2-4392-9992-414d9f4af69f";
	setAttr ".t" -type "double3" -4.673933982849121 -12.246504426002502 -4.379776120185852 ;
createNode joint -n "head_top" -p "head";
	rename -uid "a480b906-9a72-4f75-bd21-54fdfbd5897d";
	setAttr ".t" -type "double3" -4.979348182678223 -12.87928819656372 -5.32907098531723 ;
createNode joint -n "r_shoulder" -p "neck";
	rename -uid "8a45147c-0d9e-4b42-b410-41875115ec51";
	setAttr ".t" -type "double3" -13.485276699066162 7.803776860237122 0.202806293964386 ;
createNode joint -n "r_elbow" -p "r_shoulder";
	rename -uid "c657713d-1384-4e72-9f4b-3381b34ebcc5";
	setAttr ".t" -type "double3" -2.8270065784454346 13.12977522611618 32.359424233436584 ;
createNode joint -n "r_wrist" -p "r_elbow";
	rename -uid "4fabc9af-03e1-4528-b49b-10011dbcfd87";
	setAttr ".t" -type "double3" 6.65067732334137 4.00007963180542 17.45949685573578 ;
createNode joint -n "l_shoulder" -p "neck";
	rename -uid "f5503c97-bb3c-4492-9776-f59a58ce7cc8";
	setAttr ".t" -type "double3" 14.162931963801384 -5.896562337875366 3.6495912820100784 ;
createNode joint -n "l_elbow" -p "l_shoulder";
	rename -uid "1afb13bf-6f97-4e31-a228-3970d60a736a";
	setAttr ".t" -type "double3" 11.756882444024086 -16.787025332450867 16.823874786496162 ;
createNode joint -n "l_wrist" -p "l_elbow";
	rename -uid "75aa3267-7d7a-4436-843a-ca59105e7215";
	setAttr ".t" -type "double3" 10.877300798892975 -7.293391227722168 13.904658704996109 ;
