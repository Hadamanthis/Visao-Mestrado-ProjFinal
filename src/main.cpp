/*
 * main.cpp
 *
 *  Created on: Nov 4, 2016
 *      Author: geovane
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

Point COG(Mat const & img); // Funcionando corretamente
Mat contourCurvature(vector<Point> const & vecContourPoints, int step);
Mat areaFunction(vector<Point> const & contour, Point COG);
Mat triangleAreaRepresentation(vector<Point> const & contour, int ts); // Foi refeita, e finalmente desproblematizou
Mat tecnica(vector<Point> const & contour, int n, int t);
float eccentricity(vector<Point> const & contour); // Não vou testar
float det(vector<vector<float> > const & mat); // Funcionando corretamente

int main() {

	Scalar color = Scalar(0, 0, 255);

	// Tenho que obter o nome de todas as imagens e abrir todas elas
	string sbases = "/home/geovane/Bases de Imagens/LIBRAS/bases.txt";

	ifstream bases;
	bases.open(sbases.c_str());
	ofstream saida;
	saida.open("/home/geovane/Resultados/mark 2/triangleAreaRepresentation-4.txt");

	if (bases.is_open() && saida.is_open()) {
		string linha;

		int tipo = 1; // A classe da imagem que estamos processando

		while (getline(bases, linha)) {
			cout << linha << endl;

			ifstream subbase;
			subbase.open(linha.c_str());

			if (subbase.is_open()) {

				string img;

				Mat original, image, canny_output, drawing;

				Point cog;

				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;

				int threshold = 40;

				int erosion_type = MORPH_RECT;
				int erosion_size = 1;

				while (getline(subbase, img)) {

					original = imread(img, CV_LOAD_IMAGE_ANYCOLOR);

					imshow("original", original);

					Mat element = getStructuringElement(erosion_type,
							Size(2*erosion_size + 1, 2*erosion_size + 1),
							Point(erosion_size, erosion_size));

//					// Aplica morfolofia matemática
//					dilate(original, original, element);
//
//					imshow("original", original);
//
//					key = waitKey();
//
//					if (key == 27)
//						break;
//
//					erode(original, original , element);
//
//					imshow("original", original);
//
//					key = waitKey();
//
//					if (key == 27)
//						break;

					// Retirando possíveis ruidos da imagem
					blur(original, original, Size(3, 3));

					// Retorna mascara dos pixels da imagem original com intensidade > 80
					image = original > 80;

					cog = COG(image); // Calculando o centro de gravidade

					Canny(image, canny_output, threshold, 2*threshold); // Deixando as bordas

					imshow("Canny Output", canny_output);

					// Achando o contorno
					findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

					// Vetor contendo todos os pontos de todos os contornos
					vector<Point> contour;

					// Percorrendo todos os contornos
					for(int j = 0; j < contours.size(); j++) {

						// Desenhando o contorno em vermelho
						drawContours(image, contours, j, color, 1, 1, hierarchy, 0, Point(0, 0));

						// Adicionando os valores de cada contorno
						contour.insert(contour.end(), contours[j].begin(), contours[j].end());
					}

					/* As quatro técnicas */
					//Mat resultado = tecnica(contours[idx], 40, 4);
					//Mat resultado = contourCurvature(contours[0], 1);
					//Mat resultado = areaFunction(contours[0], cog);
					Mat resultado = triangleAreaRepresentation(contour, 4);

					imshow("binaria", image);

//					for (int i = 0; i < resultado.cols; i++) {
//						cout << resultado.at<float>(0, i) << " ";
//					}
//					cout << endl;

					// Normalizando o resultado entre 0 e 1
					//cv::normalize(resultado, resultado, 1, 0, NORM_MINMAX, -1, Mat());

					int hstSize = 256;
					float range[] = {0, 255};
					const float* hstRange = {range};

					bool uniform = true; bool accumulate = false;

					Mat hst;

					calcHist( &resultado, 1, 0, Mat(), hst, 1, &hstSize, &hstRange, uniform, accumulate );

					for (int i = 0; i < hst.rows; i++)
						cout << hst.at<float>(i) << " ";
					cout << endl;

					// Escrevendo o resultado no arquivo de saida
					saida << tipo << '\t';
					int contador = 1;
					for (int i = 0; i < resultado.cols; i++) {
						saida << contador++ << ':' << resultado.at<float>(0, i) << '\t';
						//saida << contador++ << ':' << resultado.at<float>(1, i) << '\t';
					}
					saida << endl;

//					int key = waitKey();
//
//					if (key == 27)
//						break;

				}

				subbase.close();
			}
			else cout << "Erro ao tentar abrir a base: " << line << endl;

			tipo++;
		}

		saida.close();
		bases.close();
	}

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
	Mat vecCurvature(1, vecContourPoints.size(), CV_32FC1 );

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

	return vecCurvature;
}

/* @brief Retorna um vetor de valores que representam a função da area do
 * contorno passado como parâmetro.
 *
 * @param contour - contorno original
 * @param COG - centro de gravidade do contorno
 * @return Vetor de pontos
 */
Mat areaFunction(vector<Point> const & contour, Point COG) {

	int size = contour.size();
	Mat func(1, size, CV_32FC1);

	// Percorrendo todos os pontos do contorno
	for (unsigned int i = 0; i < size; i++) {

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
 * @param nro_pontos - numero de pontos, igualmente espaçados, do contorno que
 * 				serão usados para o calculo da função
 *
 * @return vetor
 */
Mat triangleAreaRepresentation(vector<Point> const & contour, int ts) {

	Mat resultado;

	if (!(ts >= 1 && ts <= contour.size()/2 + 1)) return resultado;

	resultado = Mat(1, contour.size(), CV_32FC1);

	for (int i = 0; i < contour.size(); i++) {
		int anterior = (i - ts < 0) ? contour.size() + (i - ts) : i - ts;
		int posterior = (i + ts)%contour.size();

		// Matriz que descobriremos o determinante
		vector<vector<float> > mat;

		// anterior
		int p[] = {contour[anterior].x, contour[anterior].y, 1};
		vector<float> v(p, p+3);
		mat.push_back(v);

		// central
		int p2[] = {contour[i].x, contour[i].y, 1};
		vector<float> v2(p2, p2+3);
		mat.push_back(v2);

		// posterior
		int p3[] = {contour[posterior].x, contour[posterior].y, 1};
		vector<float> v3(p3, p3+3);
		mat.push_back(v3);

		float d = det(mat);

		d = (d >= 0 ? d : -1*d);

		resultado.at<float>(0, i) = (0.5*d);

	}

	return resultado;
}

Mat tecnica(vector<Point> const & contour, int n, int t) {
	Mat result(2, n, CV_32FC1);

	for (int i = 0; i < n; i++) {

		// O ponto do contorno
		int pos = int(contour.size() * (float(i)/n));
		Point p = contour[pos];

		// Os pontos vizinhos ao ponto de contorno, separados a uma distancia t
		Point p1, p2;
		int pos1;
		pos1 = pos - t >= 0 ? pos - t : contour.size() - (pos - t) ;
		p1 = contour[pos1];
		pos1 = pos + t < contour.size() ? pos + t : (pos + t) - contour.size();
		p2 = contour[pos1];

		// Agora é só calcular o gradiente e a magnitude de p1p2

		float theta = p2.x - p2.y != 0 ? tan((p2.y - p1.y)/(p2.x - p2.y)) : std::numeric_limits<float>::max();
		float magnitude = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));

		result.at<float>(0, i) = theta;
		result.at<float>(1, i) = magnitude;
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
