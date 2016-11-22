/*
 * main.cpp
 *
 *  Created on: Nov 4, 2016
 *      Author: geovane
 */

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

Point COG(Mat img);
float eccentricity(vector<Point> contour);
Mat areaFunction(vector<Point> contour, Point COG);
float det(vector<vector<float> > mat);

int main() {

	int thresh = 100;
	Scalar color = Scalar(0, 0, 255);

	Mat img = imread("c0.PNG", CV_LOAD_IMAGE_ANYCOLOR);
	Mat canny_output, drawing;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Point p = COG(img); // Calculando o centro de gravidade

	Canny(img, canny_output, thresh, 2*thresh); // Deixando as bordas

	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0)); // Achando o contorno

	//	drawing = Mat::zeros(canny_output.size(), CV_8UC3); // Iniciando a matriz de desenho dos contornos com zeros

	drawContours(img, contours, 0, color, 1, 1, hierarchy, 0, Point(0, 0)); // Desenhando o contorno em vermelho

	//Rect bb = boundingRect(contours[0]);

	//rectangle(img, bb, color);

	//circle(img, p, 1, color, -1); // Desenhando o ponto na imagem

	Mat areaf = areaFunction(contours[0], p);

	cv::normalize(areaf, areaf, 1, 0, NORM_MINMAX, -1, Mat());

	cout << "Area Function" << endl;
	for (int i = 0; i < areaf.cols; i++)
		cout << areaf.at<float>(0, i) << " ";;
	cout << endl;

	cout << "Eccentricity" << endl;
	cout << eccentricity(contours[0]) << endl;

	imshow("canny_output", canny_output);

	imshow("Lenna", img);

	waitKey();

	return 0;
}

/* @brief Retorna o Centro de Gravidade (COG) da imagem transformando-a,
 * por meio da funcao cvtColor, em uma imagem em nível de cinza.
 *
 * @param img - Mat da imagem original.
 * @return Centro de Gravidade.
 *  */
Point COG(Mat img) {
	Mat gray;

	// Testando se é necessário converter em nível de cinza
	if (img.channels() > 1) cvtColor(img, gray, CV_BGR2GRAY);
	else gray = img;

	// Calculando o COG por meio de momentos
	Moments m = moments(gray, false);
	Point p1(m.m10/m.m00, m.m01/m.m00);

	return p1;
}

float eccentricity(vector<Point> contour) {
	Rect bb = boundingRect(contour);
	//std::cout << bb.width << " " << bb.height << std::endl;
	return 1 - ((float)bb.width / bb.height);
}

/* @brief Retorna um vetor de valores que representam a função da area do
 * contorno passado como parâmetro.
 *
 * @param contour - contorno original
 * @return Vetor de pontos
 */
Mat areaFunction(vector<Point> contour, Point COG) {
	Mat func(1, contour.size(), CV_32FC1);
	int size = contour.size();

	// Percorrendo todos os pontos do contorno
	for (unsigned int i = 0; i < contour.size(); i++) {

		// Matriz que descobriremos o determinante
		vector<vector<float> > mat;

		// Ponto original
		int p[] = {contour[i].x, contour[i].y, 1};
		vector<float> v(p, p+3);
		mat.push_back(v);

		// Ponto seguinte
		int p2[] = {contour[(i+1)%size].x, contour[(i+1)%size].y, 1};
		vector<float> v2(p2, p2+3);
		mat.push_back(v2);

		// Centroide
		int p3[] = {COG.x, COG.y, 1};
		vector<float> v3(p3, p3+3);
		mat.push_back(v3);

		float d = det(mat);
		d = (d >= 0 ? d : -1*d);

		func.at<float>(0, i) = (0.5*d);

	}

	return func;
}

/* @brief Funcao que retorna um vetor de valores gerados pela função de
 * representação de área de triângulos
 *
 * @param contour - contorno original
 * @param ts - espaçamento entre os pixels consecutivos para calculo da
 * 				area do triângulo, deve ser PAR
 *
 * @return vetor
 */
Mat triangleAreaRepresentation(vector<Point> contour, int ts) {

}

/* @brief Calcula o determinante de uma matriz
 *
 *  @param mat - vetor bidimensional quadrada tratado como matriz
 *  @return determinante de mat
 */
float det(vector<vector<float> > mat) {
	int rows = mat.size();
	int cols = mat[0].size();
	float pos, neg, det = 0;

	for (int j = 0; j < cols; j++) {
		pos = neg = 1;
		for (int k = 0; k < rows; k++) {
			int new_i = k%rows;

			int new_j = (j+k)%rows;
			pos *= mat[new_i][new_j];

			new_j = j-k < 0 ? rows + (j - k) : j-k;
			neg *= mat[new_i][new_j];
		}
		det += pos - neg;
	}

	return det;
}
