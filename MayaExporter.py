import json
import uuid

class MayaExporter:
    @staticmethod
    def WriteJsonNode(file, parent, name, node, parentLoc, parentRot):
        currLoc = None
        currRot = None

        if parentLoc is None:
            parentLoc = (0,0,0)

        if parentRot is None:
            parentRot = (0,0,0)

        file.write("createNode joint -n \"" + name + "\" -p \"" + parent + "\";\n")
        file.write("\trename -uid \"" + str(uuid.uuid4()) + "\";\n")

        for subNode in node:
            if subNode == 't':
                currLoc = node[subNode]
                file.write("\tsetAttr \".t\" -type \"double3\" " + str(currLoc[0] - parentLoc[0]) + " " + str(currLoc[1] - parentLoc[1]) +" " + str(currLoc[2] - parentLoc[2]) +" ;\n")
            elif subNode == 'r':
                currRot = node[subNode]
                file.write("\tsetAttr \".r\" -type \"double3\" " + str(currLoc[0] - parentRot[0]) + " " + str(currLoc[1] - parentRot[1]) +" " + str(currRot[2] - parentRot[2]) +" ;\n")
            else:
                MayaExporter.WriteJsonNode(file, name, subNode, node[subNode], currLoc, currRot)

    @staticmethod
    def WriteToMayaAscii(filepath, data):
        fileOut = open(filepath, "w")

        frames = data.keys()
        for frame in data:
            fileOut.write("createNode transform -s -n \"" + frame + "\";\n")
            fileOut.write("\trename -uid \"" + str(uuid.uuid4()) + "\";\n")
            fileOut.write("\tsetAttr \".r\" -type \"double3\" 180 0 0 ;\n")
            for joint in data[frame]:
                MayaExporter.WriteJsonNode(fileOut, frame, joint, data[frame][joint], None, None)
        
        print("File " + filepath + " written.")
        fileOut.close()

    @staticmethod
    def WriteAtomFile(filepath, sceneFilePath, startTime, endTime, joint_data):
        fileText = MayaExporter.WriteAtomHeader(sceneFilePath, startTime, endTime)

        #baseSkeleton = MayaExporter.GetFlattenedBaseSkeleton()
        frameSkeleton = MayaExporter.GetFlattenedJoints(joint_data)

        frame = 1

        for joint in frameSkeleton.keys():
            isIkHandle = False

            #if joint == 'r_elbow' or joint == 'r_wrist' or joint == 'l_elbow' or joint == 'l_wrist' or joint == 'r_knee' or joint == 'r_ankle' or joint == 'l_knee' or joint == 'l_ankle':
            if joint == 'r_wrist'or joint == 'l_wrist' or joint == 'r_ankle' or joint == 'l_ankle':
                isIkHandle = True
            #if joint == 'r_ankle':
            #    isIkHandle = True

            frameJoint = frameSkeleton[joint]
            #baseJoint =  baseSkeleton[joint]

            translation = (frameJoint['t'][0], frameJoint['t'][1], frameJoint['t'][2]);

            #if isIkHandle:
            #    translation = frameJoint['abs']
            #else:
            #    translation = frameJoint['t']

            if isIkHandle:
                fileText += MayaExporter.WriteAtomNode(joint, frameJoint['depth'], frameJoint['children'], frame, isIkHandle, translation)

        fileOut = open(filepath, "w")
        fileOut.write(fileText)
        fileOut.close()
        print("ATOM File written: " + filepath)

        return None

    @staticmethod
    def WriteAtomHeader(sceneFilePath, startTime, endTime):
        headerText = ""
        headerText += "atomVersion 1.0;\n"
        headerText += "mayaVersion 2019;\n"
        headerText += "mayaSceneFile " + sceneFilePath + ";\n"
        #headerText += "mayaSceneFile C:/Users/imagi/Documents/maya/projects/default/scenes/sk_mannikin_margipose.0006.ma;\n"
        headerText += "timeUnit ntsc;\n"
        headerText += "linearUnit cm;\n"
        headerText += "angularUnit deg;\n"
        headerText += "startTime " + str(startTime) + ";\n"
        headerText += "endTime " + str(endTime) + ";\n"
        return headerText

    @staticmethod
    def WriteAtomAnimNode(animType, axis, position, frame, value):
        nodeText = "  anim " + animType + "." + animType + axis + " " + animType + axis + " " + str(position) + ";\n"
        nodeText += "  animData {\n"
        nodeText += "    input time;\n"
        if animType == 'rotate':
            nodeText += "    output angular;\n"
        else:
            nodeText += "    output linear;\n"
        nodeText += "    weighted 0;\n"
        nodeText += "    preInfinity constant;\n"
        nodeText += "    postInfinity constant;\n"
        nodeText += "    keys {\n"
        nodeText += "      " + str(frame) + " " + str(value) + " fixed fixed 1 0 0 0 1 0 1;\n"
        nodeText += "    }\n"
        nodeText += "  }\n"
        return nodeText;

    @staticmethod
    def WriteAtomNode(joint_name, depth, numChildren, frame, isIkHandle, translation):
        nodeText = "dagNode {\n"
        prefix = ""

        if isIkHandle:
            nodeText += "  ikHandle_" + joint_name + " 1 0;\n"
        else:
            nodeText += "  " + joint_name + " " + str(depth) + " " + str(numChildren) + ";\n"

        #translations
        nodeText += MayaExporter.WriteAtomAnimNode("translate", 'X', 0, frame, translation[0])
        nodeText += MayaExporter.WriteAtomAnimNode("translate", 'Y', 1, frame, translation[1])
        nodeText += MayaExporter.WriteAtomAnimNode("translate", 'Z', 2, frame, translation[2])

        nodeText += "}\n"

        return nodeText

    @staticmethod
    def GetBaseSkeleton():
        # create JSON file with all the joints saved
        joints_loc = {
                "root" : {
                    "t": (0,0,0),
                    "pelvis" : {
                        "t": (0, -95.645380079746246, -1.0531962472667223e-14),
                            "r_hip" : {
                                "t": (-9.8915234208106995, -0.0066176056861877441, 0),
                                "r_knee" : {
                                    "t": (-4.2926052808761597, 42.616801381111166, 1.9999999999999998),
                                    "r_ankle" : {
                                        "t": (-3.5644919872283936, 39.55141210556026, 5.9999999999999991),
                                        },
                                    },
                                },
                            "l_hip" : {
                                "t": (10.076311230659485, -0.48624202609062195, 0),
                                "l_knee" : {
                                    "t":(4.5797320753335953, 43.201074630022049, 2.0000000000000004),
                                    "l_ankle" : {
                                        "t": (2.8197032660245895, 38.723358631134062, 5.9999999999999991),
                                        },
                                    },
                                },
                    "spine_02" : {
                        "t": (1.0327458381652832, -27.163039147853851, 0),
                        "neck": {
                            "t": (-0.86338168382644664, -33.686697483062744, 5),
                            "head" : {
                                "t": (0.32834208011627197, -9.893812894821167, -2.9999999999999991),
                                "head_top":{
                                    "t": (-0.40994435548782349, -15.897387027740479, -7.3478807948841355e-16),
                                    },
                                },
                            "r_shoulder" : {
                                "t": (-19.841589450836182, 7.1010987758636475, 5.0000000000000018),
                                "r": (4.2688954223478905, 4.1274204804940444, -0.41798660163248025),
                                "r_elbow" : {
                                    "t": (-18.759292244911194, 22.884265780448843, 1.9999999999999976),
                                    "r": (-2.7704102784636153, -2.3821592983495652, 1.2715152313971241),
                                    "r_wrist" : {
                                        "t": (-18.28344273567199, 15.391905218362837, -10.999999999999998),
                                        },
                                    },
                                },
                            "l_shoulder" : {
                                "t": (17.219760805368423, 7.5286667346954346, 5),
                                "l_elbow" : {
                                    "t": (19.131802648305893, 22.008873224258423, 1.9999999999999993),
                                    "l_wrist" : {
                                        "t": (19.035491347312927, 16.03932905197145, -11.000000000000004),
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }

        return joints_loc

    @staticmethod
    def GetFlattenedBaseSkeleton():
        joints_loc = {
            "root" : {
                "t": (0,0,0),
            },
            "pelvis" : {
                "t": (0, -95.645380079746246, -1.0531962472667223e-14),
            },
            "r_hip" : {
                "t": (-9.8915234208106995, -0.0066176056861877441, 0),
            },
            "r_knee" : {
                "t": (-4.2926052808761597, 42.616801381111166, 1.9999999999999998),
            },
            "r_ankle" : {
                "t": (-3.5644919872283936, 39.55141210556026, 5.9999999999999991),
            },
            "l_hip" : {
                "t": (10.076311230659485, -0.48624202609062195, 0),
            },
            "l_knee" : {
                "t": (4.5797320753335953, 43.201074630022049, 2.0000000000000004),
            },
            "l_ankle" : {
                "t": (2.8197032660245895, 38.723358631134062, 5.9999999999999991),
            },
            "spine_02" : {
                "t": (1.0327458381652832, -27.163039147853851, 0),
            },
            "neck": {
                "t": (-0.86338168382644664, -33.686697483062744, 5),
            },
            "head" : {
                "t": (0.32834208011627197, -9.893812894821167, -2.9999999999999991),
            },
            "head_top":{
                "t": (-0.40994435548782349, -15.897387027740479, -7.3478807948841355e-16),
            },
            "r_shoulder" : {
                "t": (-19.841589450836182, 7.1010987758636475, 5.0000000000000018),
                "r": (4.2688954223478905, 4.1274204804940444, -0.41798660163248025),
            },
            "r_elbow" : {
                "t": (-18.759292244911194, 22.884265780448843, 1.9999999999999976),
                "r": (-2.7704102784636153, -2.3821592983495652, 1.2715152313971241),
            },
            "r_wrist" : {
                "t": (-18.28344273567199, 15.391905218362837, -10.999999999999998),
            },
            "l_shoulder" : {
                "t": (17.219760805368423, 7.5286667346954346, 5),
            },
            "l_elbow" : {
                "t": (19.131802648305893, 22.008873224258423, 1.9999999999999993),
            },
            "l_wrist" : {
                "t": (19.035491347312927, 16.03932905197145, -11.000000000000004),
            },
        }

        return joints_loc

    @staticmethod
    def GetFlattenedJoints(coords):
        joints_loc = {
            "root" : {
                "t": (0,0,0),
                "abs" : (0,0,0),
                "depth" : 2,
                "children" : 1
            },
            "pelvis" : {
                "t": coords[14],
                "abs" : coords[14],
                "depth" : 3,
                "children" : 3
            },
            "r_hip" : {
                "t": coords[8] - coords[14],
                "abs" : coords[8],
                "depth" : 4,
                "children" : 1
            },
            "r_knee" : {
                "t": coords[9] - coords[8],
                "abs" : coords[9],
                "depth" : 5,
                "children" : 1
            },
            "r_ankle" : {
                "t": coords[10] - coords[9],
                "abs" : coords[10],
                "depth" : 6,
                "children" : 0
            },
            "l_hip" : {
                "t": coords[11] - coords[14],
                "abs" : coords[11],
                "depth" : 4,
                "children" : 1
            },
            "l_knee" : {
                "t": coords[10] - coords[11],
                "abs" : coords[10],
                "depth" : 5,
                "children" : 1
            },
            "l_ankle" : {
                "t": coords[13] - coords[12],
                "abs" : coords[13],
                "depth" : 6,
                "children" : 0
            },
            "spine_02" : {
                "t": coords[15] - coords[14],
                "abs" : coords[15],
                "depth" : 4,
                "children" : 1
            },
            "neck": {
                "t": coords[1] - coords[15],
                "abs" : coords[1],
                "depth" : 5,
                "children" : 3
            },
            "head" : {
                "t": coords[16] - coords[1],
                "abs" : coords[16],
                "depth" : 6,
                "children" : 1
            },
            "head_top":{
                "t": coords[0] - coords[16],
                "abs" : coords[0],
                "depth" : 7,
                "children" : 0
            },
            "r_shoulder" : {
                "t": coords[0] - coords[1],
                "abs" : coords[0],
                "depth" : 6,
                "children" : 1
            },
            "r_elbow" : {
                "t": coords[3] - coords[2],
                "abs" : coords[3],
                "depth" : 7,
                "children" : 1
            },
            "r_wrist" : {
                "t": coords[4] - coords[3],
                "abs" : coords[4],
                "depth" : 8,
                "children" : 0
            },
            "l_shoulder" : {
                "t": coords[5] - coords[1],
                "abs" : coords[5],
                "depth" : 6,
                "children" : 1
            },
            "l_elbow" : {
                "t": coords[6] - coords[5],
                "abs" : coords[6],
                "depth" : 7,
                "children" : 1
            },
            "l_wrist" : {
                "t": coords[7] - coords[6],
                "abs" : coords[7],
                "depth" : 8,
                "children" : 0
            },
        }

        return joints_loc