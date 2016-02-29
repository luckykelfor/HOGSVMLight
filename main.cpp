#include <ios>
#include <fstream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <io.h>
#define YES 1
#define NO 0
#define SVMLIGHT 1
#define LIBSVM 2

#define DESCRIPTOR_FIRST_SAVED YES
#define DESCRIPTOR_SECOND_SAVED YES
//#define FEATURES_SAVED YES
#define SVMMODEL_SAVED YES
#define RANDNEG_SAVED YES
#define CENTRAL_CROP NO

//#define TRAINHOG_USEDSVM SVMLIGHT
#define TRAINHOG_USEDSVM SVMLIGHT

#if TRAINHOG_USEDSVM == SVMLIGHT
#include "svmlight/svmlight.h"
#define TRAINHOG_SVM_TO_TRAIN SVMlight
#elif TRAINHOG_USEDSVM == LIBSVM
#include "libsvm/libsvm.h"
#define TRAINHOG_SVM_TO_TRAIN libSVM
#endif


using namespace std;
using namespace cv;

//original 训练样本路径名
static string posSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/pos/";
static string originaleNegTrainingSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/neg/";
//Generated Negative samples
static string randNegTrainingSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/randneg/";

//测试样本路径名
static string testPosDir = "D:/Myworkspace/ImageData/BottlesSet/test/pos/";
static string testNegDir = "D:/Myworkspace/ImageData/BottlesSet/test/neg/";
static string testSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/test/";
//Save the parameters of the hog detector, for the time being, just the hitThreshold
static string ThresholdPath = "D:/Myworkspace/ImageData/BottlesSet/hitThreshold.dat";
//static string ThresholdPath_hard = "D:/Myworkspace/ImageData/BottlesSet/hitThreshold.dat";
//生成的难例存储路径：
static string hardExampleDir = "D:/Myworkspace/ImageData/BottlesSet/hard/";
// 存储HOG特征
static string featuresFile = "D:/Myworkspace/ImageData/BottlesSet/features.dat";


// 存储SVM模型
static string svmModelFile = "D:/Myworkspace/ImageData/BottlesSet/svmlightmodel.dat";

// 存储HOG特征描述符
static string descriptorVectorFile = "D:/Myworkspace/ImageData/BottlesSet/descriptorvector.dat";

static const Size trainingPadding = Size(0, 0); //填充值
static const Size winStride = Size(8, 8);			//窗口步进

// 字母小写转换
static string toLowerCase(const string& in) {
	string t;
	for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
		t += tolower(*i);
	}
	return t;
}

// 用于进度可视化
static void storeCursor(void) {
	printf("\033[s");
}

static void resetCursor(void) {
	printf("\033[u");
}

/**
* 保存给定的HOG特征描述符到文件
* @param descriptorVector: 待保存的HOG特征描述符矢量
* @param _vectorIndices
* @param fileName
*/
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, vector<unsigned int>& _vectorIndices, string fileName) {
	printf("保存HOG特征描述符：'%s'\n", fileName.c_str());
	string separator = " "; // 特征分隔符
	fstream File;
	File.open(fileName.c_str(), ios::out);
	if (File.good() && File.is_open()) {
		for (int feature = 0; feature < descriptorVector.size(); ++feature)
			File << descriptorVector.at(feature) << separator;	//写入特征并设置分隔符号
		File << endl;
		File.flush();
		File.close();
	}
}

/**
* 列出给定目录的所有文件，并返回字符串数组(路径+文件名)
* @param dirName: 目录名
* @param fileNames: 给定目录中找到的文件名
* @param validExtensions: 有效文件后缀规定
*/
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
	printf("扫描样本目录 %s\n", dirName.c_str());
	long hFile = 0;
	struct _finddata_t fileInfo;
	string pathName, fullfileName;
	string  tempPathName = pathName.assign(dirName);
	string  tempPathName2 = pathName.assign(dirName);

	//if ((hFile = _findfirst(pathName.assign(dirName).append("\\*").c_str(), &fileInfo)) == -1) {
	if ((hFile = _findfirst(tempPathName.append("\\*").c_str(), &fileInfo)) == -1) {
		return;
	}
	do {
		if (fileInfo.attrib&_A_SUBDIR)//文件夹跳过
			continue;
		else
		{

			int i = string(fileInfo.name).find_last_of(".");
			string tempExt = toLowerCase(string(fileInfo.name).substr(i + 1));

			if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end())
			{


				fullfileName = tempPathName2 + (string(fileInfo.name));
				//  cout<<"有效文件： '%s'\n"<< fullfileName << endl;
				fileNames.push_back((cv::String)fullfileName);



			}
			else
			{
				cout << "不是图像文件，跳过: '%s'\n" << fileInfo.name << endl;
			}


		}

	} while (_findnext(hFile, &fileInfo) == 0);
	_findclose(hFile);
	return;

}
/**
* @file:   main.cpp
* @author: Jan Hendriks (dahoc3150 [at] gmail.com)
* @date:   Created on 2. Dezember 2012
* @brief:  Example program on how to train your custom HOG detecting vector
* for use with openCV <code>hog.setSVMDetector(_descriptor)</code>;
*
* Copyright 2015 Jan Hendriks
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*
* For the paper regarding Histograms of Oriented Gradients (HOG), @see http://lear.inrialpes.fr/pubs/2005/DT05/
* You can populate the positive samples dir with files from the INRIA person detection dataset, @see http://pascal.inrialpes.fr/data/human/
* This program uses SVMlight as machine learning algorithm (@see http://svmlight.joachims.org/), but is not restricted to it
* Tested in Ubuntu Linux 64bit 12.04 "Precise Pangolin" with openCV 2.3.1, SVMlight 6.02, g++ 4.6.3
* and standard HOG settings, training images of size 64x128px.
*
* What this program basically does:
* 1. Read positive and negative training sample image files from specified directories
* 2. Calculate their HOG features and keep track of their classes (pos, neg)
* 3. Save the feature map (vector of vectors/matrix) to file system
* 4. Read in and pass the features and their classes to a machine learning algorithm, e.g. SVMlight
* 5. Train the machine learning algorithm using the specified parameters
* 6. Use the calculated support vectors and SVM model to calculate a single detecting descriptor vector
* 7. Dry-run the newly trained custom HOG descriptor against training set and against camera images, if available
*
* Build by issuing:
* g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF main.o.d -o main.o main.cpp
* gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_learn.o.d -o svmlight/svm_learn.o svmlight/svm_learn.c
* gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_hideo.o.d -o svmlight/svm_hideo.o svmlight/svm_hideo.c
* gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_common.o.d -o svmlight/svm_common.o svmlight/svm_common.c
* g++ `pkg-config --cflags opencv` -o trainhog main.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`
*
* Warning:
* Be aware that the program may consume a considerable amount of main memory, hard disk memory and time, dependent on the amount of training samples.
* Also be aware that (esp. for 32bit systems), there are limitations for the maximum file size which may take effect when writing the features file.
*
* Terms of use:
* This program is to be used as an example and is provided on an "as-is" basis without any warranties of any kind, either express or implied.
* Use at your own risk.
* For used third-party software, refer to their respective terms of use and licensing.
*/

// <editor-fold defaultstate="collapsed" desc="Definitions">
/* Parameter and macro definitions */




/**
* This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
* @param imageFilename file path of the image file to read and calculate feature vector from
* @param descriptorVector the returned calculated feature vector<float> ,
*      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. Mat<float>)
* @param hog HOGDescriptor containin HOG settings
*/
static void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, HOGDescriptor& hog) {
	/** for imread flags from openCV documentation,
	* @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat imread(const string& filename, int flags)
	* @note If you get a compile-time error complaining about following line (esp. imread),
	* you either do not have a current openCV version (>2.0)
	* or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
	*/
	Mat imageData = imread(imageFilename, 0);


	if (CENTRAL_CROP)
	{
		if (imageData.size() == Size(96, 160))
			imageData = imageData(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
		else
			resize(imageData, imageData, Size(64, 128));
	}

	if (imageData.empty()) {
		featureVector.clear();
		printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
		return;
	}
	// Check for mismatching dimensions
	// if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height) {
	//     featureVector.clear();
	//     printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), imageData.cols, imageData.rows, hog.winSize.width, hog.winSize.height);
	//     return;
	// }
	vector<Point> locations;
	hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
	// imageData.release(); // Release the image again after features are extracted
}

/**
* Shows the detections in the image
* @param found vector containing valid detection rectangles
* @param imageData the image in which the detections are drawn
*/

/**
* Shows the detections in the image
* @param found vector containing valid detection rectangles
* @param imageData the image in which the detections are drawn
*/
static void showDetections(const vector<Rect>& found, Mat& imageData) {
	vector<Rect> found_filtered;
	size_t i, j;
	for (i = 0; i < found.size(); ++i) {
		Rect r = found[i];
		for (j = 0; j < found.size(); ++j)
		if (j != i && (r & found[j]) == r)
			break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	for (i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
	}
}

/**
* Test the trained detector against the same training set to get an approximate idea of the detector.
* Warning: This does not allow any statement about detection quality, as the detector might be overfitting.
* Detector quality must be determined using an independent test set.
* @param hog
*/
static void detectOnSet(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames) {
	unsigned int truePositives = 0;
	unsigned int trueNegatives = 0;
	unsigned int falsePositives = 0;
	unsigned int falseNegatives = 0;
	vector<Point> foundDetection;
	// Walk over positive training samples, generate images and detect
	for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
		const Mat imageData = imread(*posTrainingIterator, 0);
		hog.detect(imageData, foundDetection, abs(hitThreshold), winStride, trainingPadding);
		if (foundDetection.size() > 0) {
			++truePositives;
			falseNegatives += foundDetection.size() - 1;
		}
		else {
			++falseNegatives;
		}
	}
	// Walk over negative training samples, generate images and detect
	for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
		const Mat imageData = imread(*negTrainingIterator, 0);
		hog.detect(imageData, foundDetection, abs(hitThreshold), winStride, trainingPadding);
		if (foundDetection.size() > 0) {
			falsePositives += foundDetection.size();
		}
		else {
			++trueNegatives;
		}
	}

	printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
}

static void detectTest(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& testImages)
{
	vector<Rect> foundDetection;
	for (vector<string>::const_iterator posTrainingIterator = testImages.begin(); posTrainingIterator != testImages.end(); ++posTrainingIterator)
	{

		Mat imageData = imread(*posTrainingIterator);

		hog.detectMultiScale(imageData, foundDetection, abs(hitThreshold), winStride, trainingPadding);

		showDetections(foundDetection, imageData);
		
		namedWindow("HH",0);
		imshow("HH", imageData);
		waitKey();

	}
}

/*Hard Example 难例二次训练*/

static long findHardExmaple(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& negFileNames, const string hardExampleDir)
{

	//在Neg中检测出来的都是误报的，将多尺度检测出来的窗口作为难例。
	char saveName[256];
	long hardExampleCount = 0;
	vector<Rect> foundDetection;
	// Walk over negative training samples, generate images and detect
	for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator)
	{
		const Mat imageData = imread(*negTrainingIterator);
		hog.detectMultiScale(imageData, foundDetection, abs(hitThreshold), winStride, trainingPadding);
		//遍历从图像中检测出来的矩形框，得到hard example
		for (int i = 0; i < foundDetection.size(); i++)
		{
			//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部
			Rect r = foundDetection[i];
			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > imageData.cols)
				r.width = imageData.cols - r.x;
			if (r.y + r.height > imageData.rows)
				r.height = imageData.rows - r.y;

			//将矩形框保存为图片，就是Hard Example
			Mat hardExampleImg = imageData(r);//从原图上截取矩形框大小的图片
			resize(hardExampleImg, hardExampleImg, Size(64, 64));//将剪裁出来的图片缩放为64*128大小
			sprintf(saveName, "hardexample%09ld.jpg", hardExampleCount++);//生成hard example图片的文件名
			imwrite(hardExampleDir + saveName, hardExampleImg);//保存文件


			//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
			//r.x += cvRound(r.width*0.1);
			//r.width = cvRound(r.width*0.8);
			//r.y += cvRound(r.height*0.07);
			//r.height = cvRound(r.height*0.8);
			// rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 3);

		}



	}
	printf("Hard Example Detected Done.\n");
	return hardExampleCount;


}
/**
* Test detection with custom HOG description vector
* @param hog
* @param hitThreshold threshold value for detection
* @param imageData
*/
static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData)
{
	vector<Rect> found;
	Size padding(Size(1, 1));
	Size winStride(Size(8, 8));
	hog.detectMultiScale(imageData, found, abs(hitThreshold), winStride, padding);
	showDetections(found, imageData);
}
// </editor-fold>

/**
* Main program entry point
* @param argc unused
* @param argv unused
* @return EXIT_SUCCESS (0) or EXIT_FAILURE (1)
*/
int main() {

	// <editor-fold defaultstate="collapsed" desc="Init">
	HOGDescriptor hog; // Use standard parameters here
	hog.winSize = Size(64, 64); // Default training images size as used in paper
	// Get the files to train from somewhere
	static vector<string> posTrainingSamples,
		negTrainingSamples,
		randNegTrainingSamples,
		testNegSamples,
		testPosSamples,
		testSamples,
		validExtensions;

	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("bmp");
	// </editor-fold>

	// <editor-fold defaultstate="collapsed" desc="Read image files">
	getFilesInDirectory(posSamplesDir, posTrainingSamples, validExtensions);
	getFilesInDirectory(originaleNegTrainingSamplesDir, negTrainingSamples, validExtensions);
	getFilesInDirectory(testPosDir, testPosSamples, validExtensions);
	getFilesInDirectory(testNegDir, testNegSamples, validExtensions);
	getFilesInDirectory(testSamplesDir, testSamples, validExtensions);

	//Randomly generate 10 negative sample of size 64*128 for each original negative sample in /neg folder. Totally we weill ge 1,2180 neg samples

	//图片大小应该能能至少包含一个64*128的窗口
	if (RANDNEG_SAVED == NO)
	{

		char saveName[256];
		int CropImageCount = 0;
		for (vector<string>::iterator i = negTrainingSamples.begin(); i<negTrainingSamples.end(); i++)
		{
			Mat src = imread(*i);
			if (src.cols >= 64 && src.rows >= 64)
			{
				srand(time(NULL));//设置随机数种子

				//从每张图片中随机裁剪10个64*128大小的不包含人的负样本
				for (int i = 0; i<10; i++)
				{
					int x = (rand() % (src.cols - 64)); //左上角x坐标
					int y = (rand() % (src.rows - 64)); //左上角y坐标
					//cout<<x<<","<<y<<endl;
					Mat imgROI = src(Rect(x, y, 64, 64));
					sprintf(saveName, "noperson%06d.jpg", ++CropImageCount);//生成裁剪出的负样本图片的文件名
					imwrite(randNegTrainingSamplesDir + saveName, imgROI);//保存文件
				}
			}
		}
	}


	getFilesInDirectory(randNegTrainingSamplesDir, randNegTrainingSamples, validExtensions);
	/// Retrieve the descriptor vectors from the samples
	unsigned long overallSamples = posTrainingSamples.size() + randNegTrainingSamples.size();

	// Calculate HOG features and save to file
	// Make sure there are actually samples to train
	if (overallSamples == 0) {
		printf("No training sample files found, nothing to do!\n");
		return EXIT_SUCCESS;
	}

	/// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
	setlocale(LC_ALL, "C"); // Do not use the system locale
	setlocale(LC_NUMERIC, "C");
	setlocale(LC_ALL, "POSIX");

	/**
	* Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
	* @NOTE: If you split these steps into separate steps:
	* 1. calculating features into memory (e.g. into a cv::Mat or vector< vector<float> >),
	* 2. saving features to file / directly inject from memory to machine learning algorithm,
	* the program may consume a considerable amount of main memory
	*/
#if DESCRIPTOR_FIRST_SAVED == NO

	printf("Reading files, generating HOG features and save them to file '%s':\n", featuresFile.c_str());
	float percent;

	fstream File;
	File.open(featuresFile.c_str(), ios::out);
	if (File.good() && File.is_open()) {
#if TRAINHOG_USEDSVM == SVMLIGHT
		// Remove following line for libsvm which does not support comments
		File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile.c_str() << endl;
#endif
		// Iterate over sample images
		for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile) {
			storeCursor();
			vector<float> featureVector;
			// Get positive or negative sample image file path
			const string currentImageFile = (currentFile < posTrainingSamples.size() ? posTrainingSamples.at(currentFile) : randNegTrainingSamples.at(currentFile - posTrainingSamples.size()));
			// Output progress
			if ((currentFile + 1) % 10 == 0 || (currentFile + 1) == overallSamples) {
				percent = ((currentFile + 1) * 100 / overallSamples);
				printf("%5lu (%3.0f%%):\tFile '%s'", (currentFile + 1), percent, currentImageFile.c_str());
				fflush(stdout);
				resetCursor();
			}
			// Calculate feature vector from current image file
			calculateFeaturesFromInput(currentImageFile, featureVector, hog);
			if (!featureVector.empty()) {
				/* Put positive or negative sample class to file,
				* true=positive, false=negative,
				* and convert positive class to +1 and negative class to -1 for SVMlight
				*/
				File << ((currentFile < posTrainingSamples.size()) ? "+1" : "-1");
				// Save feature vector components
				for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
					File << " " << (feature + 1) << ":" << featureVector.at(feature);
				}
				File << endl;
			}
		}
		printf("\n");
		File.flush();
		File.close();
	}
	else {
		printf("Error opening file '%s'!\n", featuresFile.c_str());
		return EXIT_FAILURE;
	}




	printf("Calling %s\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
	TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
	TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
	printf("Training done, saving model file!\n");
	TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);


	printf("SVMlight生成单个HOG特征矢量\n");
	vector<float> descriptorVector;
	vector<unsigned int> descriptorVectorIndices;
	TRAINHOG_SVM_TO_TRAIN::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
	saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
	// Detector detection tolerance threshold  检测子的检测容差
	double hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();//应该保存下来的Theshold
	ofstream writeThreshold(ThresholdPath.c_str(), ios::out);
	writeThreshold << hitThreshold;
	writeThreshold.close();

	//寻找难例
	printf("Find Hard Examples.\n");
	int hardExmapleCount = findHardExmaple(hog, hitThreshold, negTrainingSamples, hardExampleDir);
	printf("Hard Example Count:%ld\n", hardExmapleCount);

	//追加难例，重新生成feature.mat(相当于增加example)
	ofstream addingHardExampleFeature(featuresFile.c_str(), ios::app);

	//得到难例列表
	vector<string>hardExampleImages;
	vector<string>::iterator hardExample;
	getFilesInDirectory(hardExampleDir, hardExampleImages, validExtensions);
	for (hardExample = hardExampleImages.begin(); hardExample<hardExampleImages.end(); hardExample++)
	{
		vector<float> featureVector;
		calculateFeaturesFromInput(*hardExample, featureVector, hog);
		if (!featureVector.empty()) {
			/* Put positive or negative sample class to file,
			* true=positive, false=negative,
			* and convert positive class to +1 and negative class to -1 for SVMlight
			*/
			addingHardExampleFeature << "-1";//all negative
			// Save feature vector components
			for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
				addingHardExampleFeature << " " << (feature + 1) << ":" << featureVector.at(feature);
			}
			addingHardExampleFeature << endl;
		}
	}
	addingHardExampleFeature.close();



	//重新训练
	printf("Calling %s (Having added hard examples)\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
	TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
	TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
	printf("Second Training done, saving model file!\n");
	TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);





	//重新生成descriptor 向量
	printf("SVMlight生成单个HOG特征矢量\n");
	vector<float> descriptorVector_1;
	vector<unsigned int>descriptorVectorIndices_1;
	SVMlight::getInstance()->getSingleDetectingVector(descriptorVector_1, descriptorVectorIndices_1);
	saveDescriptorVectorToFile(descriptorVector_1, descriptorVectorIndices_1, descriptorVectorFile);



	hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();//应该保存下来的Theshold

	hog.setSVMDetector(descriptorVector_1);

	ofstream writeThreshold_hard(ThresholdPath.c_str(), ios::out);
	writeThreshold_hard << hitThreshold;
	writeThreshold_hard.close();

	//在训练集上进行行人检测
	cout << "Detect People on the training set. Just to roughly evluate this detector.\n";
	detectOnSet(hog, hitThreshold, posTrainingSamples, negTrainingSamples);

	return 0;


#else //If the first trained descriptor saved, just load the saved svmModel from file.


#if DESCRIPTOR_SECOND_SAVED == NO

	cout << "Loading SVM Model from file. May take some time...\n";
	TRAINHOG_SVM_TO_TRAIN::getInstance()->loadModelFromFile(svmModelFile);
	cout << "Load SVM Model Done.\n";

	printf("SVMlight生成单个HOG特征矢量\n");
	vector<float> descriptorVector;
	vector<unsigned int> descriptorVectorIndices;
	TRAINHOG_SVM_TO_TRAIN::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
	saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);
	// Detector detection tolerance threshold  检测子的检测容差
	double hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();//应该保存下来的Theshold
	ofstream writeThreshold(ThresholdPath.c_str(), ios::out);
	writeThreshold << hitThreshold;
	writeThreshold.close();


	printf("Find Hard Examples.\n");
	int hardExmapleCount = findHardExmaple(hog, hitThreshold, negTrainingSamples, hardExampleDir);
	printf("Hard Example Count:%ld\n", hardExmapleCount);

	//追加难例，重新生成feature.mat(相当于增加example)
	ofstream addingHardExampleFeature(featuresFile.c_str(), ios::app);

	//得到难例列表
	vector<string>hardExampleImages;
	vector<string>::iterator hardExample;
	getFilesInDirectory(hardExampleDir, hardExampleImages, validExtensions);
	for (hardExample = hardExampleImages.begin(); hardExample<hardExampleImages.end(); hardExample++)
	{
		vector<float> featureVector;
		calculateFeaturesFromInput(*hardExample, featureVector, hog);
		if (!featureVector.empty()) {
			/* Put positive or negative sample class to file,
			* true=positive, false=negative,
			* and convert positive class to +1 and negative class to -1 for SVMlight
			*/
			addingHardExampleFeature << "-1";//all negative
			// Save feature vector components
			for (unsigned int feature = 0; feature < featureVector.size(); ++feature) {
				addingHardExampleFeature << " " << (feature + 1) << ":" << featureVector.at(feature);
			}
			addingHardExampleFeature << endl;
		}
	}
	addingHardExampleFeature.close();



	//重新训练
	printf("Calling %s (Having added hard examples)\n", TRAINHOG_SVM_TO_TRAIN::getInstance()->getSVMName());
	TRAINHOG_SVM_TO_TRAIN::getInstance()->read_problem(const_cast<char*> (featuresFile.c_str()));
	TRAINHOG_SVM_TO_TRAIN::getInstance()->train(); // Call the core libsvm training procedure
	printf("Second Training done, saving model file!\n");
	TRAINHOG_SVM_TO_TRAIN::getInstance()->saveModelToFile(svmModelFile);





	//重新生成descriptor 向量
	printf("SVMlight生成单个HOG特征矢量\n");
	vector<float> descriptorVector_1;
	vector<unsigned int>descriptorVectorIndices_1;
	SVMlight::getInstance()->getSingleDetectingVector(descriptorVector_1, descriptorVectorIndices_1);
	saveDescriptorVectorToFile(descriptorVector_1, descriptorVectorIndices_1, descriptorVectorFile);



	hitThreshold = TRAINHOG_SVM_TO_TRAIN::getInstance()->getThreshold();//应该保存下来的Theshold

	hog.setSVMDetector(descriptorVector_1);

	ofstream writeThreshold_hard(ThresholdPath.c_str(), ios::out);
	writeThreshold_hard << hitThreshold;
	writeThreshold_hard.close();


#else // Just load the final descriptor

	ifstream  fin_detector(descriptorVectorFile.c_str());
	float temp;
	double hitThreshold;
	vector<float> descriptorVector;//3781维的检测器参数
	while (!fin_detector.eof())
	{
		fin_detector >> temp;
		descriptorVector.push_back(temp);//放入检测器数组
	}

	descriptorVector.pop_back();
	ifstream readThreshold(ThresholdPath.c_str(), ios::in);
	readThreshold >> hitThreshold;
	readThreshold.close();
	// const double hitThreshold
	cout << "The size of Descripto Vector:" << descriptorVector.size() << endl;
	// Set our custom detecting vector  设置为我们自己的支持向量（检测子向量）
	hog.setSVMDetector(descriptorVector);
#endif


#endif
	//再次在training集上测试

	cout << "Detect People on the training set. Just to roughly evluate this detector.\n";
	//detectOnSet(hog, hitThreshold, posTrainingSamples, negTrainingSamples);

	//    detectOnSet(hog,hitThreshold,posTrainingSamples,negTrainingSamples);
	detectTest(hog, hitThreshold, testSamples);
	return EXIT_SUCCESS;
}


/*the default descriptor on its own dataset:true positive(2126/2416) true negative(1139/1218) false positive(100) false Negative(2650)*/
/*After two rounds of training, the result is: (2025/2416) (1138/1218) falsePos(112) falseNeg(1616)*/
