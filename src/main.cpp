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
#include <limits>

using namespace std;
using namespace cv;

Point COG(Mat const & img);
Mat contourCurvature(vector<Point> const & vecContourPoints, int step);
float eccentricity(vector<Point> const & contour);
Mat areaFunction(vector<Point> const & contour, Point COG);
Mat triangleAreaRepresentation(vector<Point> const & contour, int ts);
float det(vector<vector<float> > const & mat);

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

	Mat curvature = contourCurvature(contours[0], 1);

	cv::normalize(curvature, curvature, 1, 0, NORM_MINMAX, -1, Mat());

	cout << "Contour Curvature" << endl;
	for (int i = 0; i < curvature.cols; i++)
		cout << curvature.at<float>(0, i) << " ";
	cout << endl;

	Mat areaF = areaFunction(contours[0], p);

	cv::normalize(areaF, areaF, 1, 0, NORM_MINMAX, -1, Mat());

	cout << "Area Function" << endl;
	for (int i = 0; i < areaF.cols; i++)
		cout << areaF.at<float>(0, i) << " ";
	cout << endl;

	Mat triangleAreaF = triangleAreaRepresentation(contours[0], 2);

	cv::normalize(triangleAreaF, triangleAreaF, 1, 0, NORM_MINMAX, -1, Mat());

	cout << "Triangle Area Representation" << endl;
	for (int i = 0; i < triangleAreaF.cols; i++)
		cout << triangleAreaF.at<float>(0, i) << " ";
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
Point COG(Mat const & img) {
	Mat gray;

	// Testando se é necessário converter em nível de cinza
	if (img.channels() > 1) cvtColor(img, gray, CV_BGR2GRAY);
	else gray = img;

	// Calculando o COG por meio de momentos
	Moments m = moments(gray, false);
	Point p1(m.m10/m.m00, m.m01/m.m00);

	return p1;
}

float eccentricity(vector<Point> const & contour) {
	Rect bb = boundingRect(contour);
	//std::cout << bb.width << " " << bb.height << std::endl;
	return 1 - ((float)bb.width / bb.height);
}

Mat contourCurvature(vector<Point> const & vecContourPoints, int step) {
	Mat vecCurvature(1, vecContourPoints.size(), CV_32FC1 ); // Terminar de essa função

	if (vecContourPoints.size() < step)
		return vecCurvature;

	Point2f frontToBack = vecContourPoints.front() - vecContourPoints.back();

	bool isClosed = ((int)std::max(std::abs(frontToBack.x), std::abs(frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < vecContourPoints.size(); i++ )
	{
		const cv::Point2f& pos = vecContourPoints[i];

		int maxStep = step;
		if (!isClosed)
		{
			std::min(std::min(step, i), (int)vecContourPoints.size()-1-i);
			if (maxStep == 0)
			{
				vecCurvature.at<float>(0, i) = std::numeric_limits<float>::max();
				continue;
			}
		}


		int iminus = i-maxStep;
		int iplus = i+maxStep;
		pminus = vecContourPoints[iminus < 0 ? iminus + vecContourPoints.size() : iminus];
		pplus = vecContourPoints[iplus > vecContourPoints.size() ? iplus - vecContourPoints.size() : iplus];


		f1stDerivative.x =   (pplus.x -        pminus.x) / (iplus-iminus);
		f1stDerivative.y =   (pplus.y -        pminus.y) / (iplus-iminus);
		f2ndDerivative.x = (pplus.x - 2*pos.x + pminus.x) / ((iplus-iminus)/2*(iplus-iminus)/2);
		f2ndDerivative.y = (pplus.y - 2*pos.y + pminus.y) / ((iplus-iminus)/2*(iplus-iminus)/2);

		double curvature2D;
		double divisor = pow(f1stDerivative.x, 2) + pow(f1stDerivative.y, 2);
		if ( std::abs(divisor) != 0 ) {
			curvature2D =  std::abs(f2ndDerivative.y*f1stDerivative.x - f2ndDerivative.x*f1stDerivative.y) /
					pow(divisor, 3.0/2.0 )  ;
		}
		else
			curvature2D = std::numeric_limits<float>::max();

		vecCurvature.at<float>(0, i) = curvature2D;

	}

	cout << std::numeric_limits<double>::max() << endl;

	return vecCurvature;
}

/* @brief Retorna um vetor de valores que representam a função da area do
 * contorno passado como parâmetro.
 *
 * @param contour - contorno original
 * @return Vetor de pontos
 */
Mat areaFunction(vector<Point> const & contour, Point COG) {
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
Mat triangleAreaRepresentation(vector<Point> const & contour, int ts) {

	Mat result;

	if (!(ts >= 1 && ts <= contour.size()/2 + 1)) return result;

	result = Mat(1, contour.size(), CV_32FC1);

	for (unsigned int i = 0; i < contour.size(); i++) {
		int anterior = i - ts < 0 ? contour.size() - 1 + (i - ts) : i - ts;
		int posterior = (i + ts)%contour.size();

		// Matriz que descobriremos o determinante
		vector<vector<float> > mat;

		// Ponto original
		int p[] = {contour[anterior].x, contour[anterior].y, 1};
		vector<float> v(p, p+3);
		mat.push_back(v);

		// Ponto seguinte
		int p2[] = {contour[i].x, contour[i].y, 1};
		vector<float> v2(p2, p2+3);
		mat.push_back(v2);

		// Centroide
		int p3[] = {contour[posterior].x, contour[posterior].y, 1};
		vector<float> v3(p3, p3+3);
		mat.push_back(v3);

		float d = det(mat);
		d = (d >= 0 ? d : -1*d);

		result.at<float>(0, i) = (0.5*d);

	}

	return result;
}

/* @brief Calcula o determinante de uma matriz
 *
 *  @param mat - vetor bidimensional quadrada tratado como matriz
 *  @return determinante de mat
 */
float det(vector<vector<float> > const & mat) {
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
