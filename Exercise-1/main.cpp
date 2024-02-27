#include <iostream>
#include <fstream>
#include <array>

#include "Eigen.h"
#include "VirtualSensor.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = 0; //width*height

	// TODO: Determine number of valid faces
	unsigned nFaces = 0;//2 * (width - 1) * (height - 1) 


	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;


	// TODO: save vertices
	std::stringstream ss;
	ss << "# list of vertices" << std::endl;
	ss << "# X Y Z R G B A" << std::endl;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Vector4f position = vertices[y * width + x].position;
			if (position.x() == MINF)
				position = Vector4f(0, 0, 0, 0);

			Vector4uc color = vertices[y * width + x].color;
			ss << position.x() << " "
				<< position.y() << " " 
				<< position.z() << " "
				<< static_cast<int>(color.x()) << " "
				<< static_cast<int>(color.y()) << " "
				<< static_cast<int>(color.z()) << " "
				<< static_cast<int>(color.w()) << std::endl;
			nVertices++;
		}
	}

	// TODO: save valid faces
	ss << "# list of faces" << std::endl;
	ss << "# nVerticesPerFace idx0 idx1 idx2" << std::endl;

	for (unsigned int i = 0; i < height - 1; ++i) {
		for (unsigned int j = 0; j < width - 1; ++j) {
			int idx = i * width + j;

			Vector4f idx0 = vertices[idx].position;
			Vector4f idx1 = vertices[idx + 1].position;
			Vector4f idx2 = vertices[idx + width].position;
			Vector4f idx3 = vertices[idx + width + 1].position;

			if (idx0.x() == MINF || idx1.x() == MINF || idx2.x() == MINF || idx3.x() == MINF)
				continue;

			if ((idx2 - idx1).squaredNorm() > edgeThreshold)
				continue;

			if ((idx0 - idx2).squaredNorm() <= edgeThreshold && (idx1 - idx0).squaredNorm() <= edgeThreshold) {
				ss << "3 " << idx << " " << idx + width << " " << idx + 1 << std::endl;
				nFaces++;
			}
			if ((idx2 - idx3).squaredNorm() <= edgeThreshold && (idx3 - idx1).squaredNorm() <= edgeThreshold) {
				ss << "3 " << idx + width << " " << idx + width + 1 << " " << idx + 1 << std::endl;
				nFaces++;
			}
		}
	}

	outFile << nVertices << " " << nFaces << " 0" << std::endl;
	outFile << ss.str() << std::endl;


	// close file
	outFile.close();

	return true;
}

int main()
{
	// Make sure this path points to the data folder
	std::string filenameIn = "C:/Users/jacop/Desktop/TUM/3D_Scanning_and_motion_capture/Exercises/Data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();

		float fX = depthIntrinsics(0, 0);
		float fY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);

		// compute inverse depth extrinsics
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();


		Matrix4f depthEctrinsics = sensor.GetDepthExtrinsics();
		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		Vertex* vertices = new Vertex[sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight()];


		for (unsigned int uy = 0; uy < sensor.GetDepthImageHeight(); ++uy) { 
			for (unsigned int ux = 0; ux < sensor.GetDepthImageWidth(); ++ux) {
				int idx = uy * sensor.GetDepthImageWidth() + ux;
				float depth = depthMap[idx];

				Vector3f Pippo{ float(ux), float(uy), 1.0 };
				
				if (depth == MINF) {
					vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
					vertices[idx].color = Vector4uc(0, 0, 0, 0);
				}
				else {
					Vector3f Pluto = depthIntrinsicsInv * Pippo * depth;
					Vector4f World = trajectoryInv * depthExtrinsicsInv * Vector4f(Pluto[0], Pluto[1], Pluto[2], 1.0f);
					
					vertices[idx].position = World;

					int idxCol = 4 * idx;
					vertices[idx].color = Vector4uc(static_cast<unsigned char>(colorMap[idxCol]), static_cast<unsigned char>(colorMap[idxCol+1]), static_cast<unsigned char>(colorMap[idxCol+2]), static_cast<unsigned char>(colorMap[idxCol+3]));
				}
			}
		}
		
		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}
		// free mem
		delete[] vertices;
	}


	return 0;
}