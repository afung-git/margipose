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

