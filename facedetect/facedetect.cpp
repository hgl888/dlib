#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "time.h"


using namespace dlib;
using namespace std;
//using namespace cv; ��dlib�������ռ��г�ͻ

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	image_window win, win_faces;
	//string face_cascade_name = "koestinger_cascade_aflw_lbp.xml";
	string face_cascade_name = "haarcascade_frontalface_default.xml";
	//����ʹ�õ�LBP��������ٶȽ�haar������ٶȿ죬û�еĻ�ʹ��opencv�Դ���haar���������Ҳ����
	cv::CascadeClassifier face_cascade;
	face_cascade.load(face_cascade_name);

	shape_predictor sp;
	string shape_model = "shape_predictor_68_face_landmarks.dat";
	
	deserialize(shape_model) >> sp;

	string img_path = "timg.jpg";
	array2d<rgb_pixel> img;
	load_image(img, img_path);
	cv::Mat face = cv::imread(img_path);

	std::vector<cv::Rect> faces;
	cv::Mat face_gray;
	cvtColor(face, face_gray, CV_BGR2GRAY);  //rgb����ת��Ϊ�Ҷ�����
	equalizeHist(face_gray, face_gray);   //ֱ��ͼ���⻯
	face_cascade.detectMultiScale(face_gray, faces, 1.2, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(20, 20));
	dlib::rectangle det;
	//��opencv��⵽�ľ���ת��Ϊdlib��Ҫ�����ݽṹ������û���жϼ�ⲻ�����������
	det.set_left(faces[0].x);
	det.set_top(faces[0].y);
	det.set_right(faces[0].x + faces[0].width);
	det.set_bottom(faces[0].y + faces[0].height);
	// Now we will go ask the shape_predictor to tell us the pose of
	// each face we detected.
	std::vector<full_object_detection> shapes;

	full_object_detection shape = sp(img, det);
	cout << "number of parts: " << shape.num_parts() << endl;
	cout << "pixel position of first part:  " << shape.part(0) << endl;
	cout << "pixel position of second part: " << shape.part(1) << endl;

	shapes.push_back(shape);

	// Now let's view our face poses on the screen.
	win.clear_overlay();
	win.set_image(img);
	win.add_overlay(render_face_detections(shapes));

	// We can also extract copies of each face that are cropped, rotated upright,
	// and scaled to a standard size as shown here:
	dlib::array<array2d<rgb_pixel> > face_chips;
	extract_image_chips(img, get_face_chip_details(shapes), face_chips);
	win_faces.set_image(tile_images(face_chips));
	char pause;
	cin >> pause;

	return 0;
}