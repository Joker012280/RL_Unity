//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Collections.Generic;
using System.Text;
using RosMessageGeneration;

namespace RosMessageTypes.FileServer
{
    public class GetBinaryFileRequest : Message
    {
        public const string RosMessageName = "file_server/GetBinaryFile";

        public string name;

        public GetBinaryFileRequest()
        {
            this.name = "";
        }

        public GetBinaryFileRequest(string name)
        {
            this.name = name;
        }
        public override List<byte[]> SerializationStatements()
        {
            var listOfSerializations = new List<byte[]>();
            listOfSerializations.Add(SerializeString(this.name));

            return listOfSerializations;
        }

        public override int Deserialize(byte[] data, int offset)
        {
            var nameStringBytesLength = DeserializeLength(data, offset);
            offset += 4;
            this.name = DeserializeString(data, offset, nameStringBytesLength);
            offset += nameStringBytesLength;

            return offset;
        }

    }
}
