// for std
#include <iostream>
#include <string>
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;

typedef vector<cv::DMatch> PairMatches;
// get sift corresponding points from two input images
int getCorrespondingPoints( const string& image1, const string& image2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2 );

// intrinsic parameters of camera
double cx = 2292.0;
double cy = 1299.5;
double fx = 3941.5;
double fy = 3941.5;

int main( int argc, char** argv )
{
    if (argc != 3)
    {
        cout<<"Usage: should indicate img1, img2"<<endl;
        exit(1);
    }
    
    // read images and get corresponding points
    string filename1 = argv[1];
    string filename2 = argv[2];
    vector<cv::Point2f> pts1, pts2;
    if ( getCorrespondingPoints( filename1, filename2, pts1, pts2 ) == false )
    {
        cout<<"not enough match points！"<<endl;
        return 0;
    }
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3(linearSolver);
    // using L-M algorithm
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg( block_solver );
    
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose( false );
    
    // add vertex camera pose
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // fix the first one
        v->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( v );
    }
    // add world points, suppose the depth is 1, use the key points of image 1 as world points
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        double z = 1;
        double x = ( pts1[i].x - cx ) * z / fx; 
        double y = ( pts1[i].y - cy ) * z / fy; 
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x,y,z) );
        optimizer.addVertex( v );
    }
    
    // prepare the camera intrinsic parameters
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
    
    // add edges, the first frame
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    // the second frame
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    
    cout<<"starting optimization ..."<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    cout<<"optimization finished! "<<endl;
    
    //output the extrinsic parameter of the second frame
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();
    cout<<"Pose="<<endl<<pose.matrix()<<endl;
    
    //output the world points
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<<i+2<<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout<<pos(0)<<","<<pos(1)<<","<<pos(2)<<endl;
    }
    
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else 
        {
            inliers++;
        }
    }
    
    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size()+pts2.size()<<endl;
    optimizer.save("ba.g2o");
    return 0;
}


int getCorrespondingPoints( const string& image1, const string& image2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2 )
{
    // read input images
    cv::Mat const& img1 = cv::imread(image1, 1);
    cv::Mat const& img2 = cv::imread(image2, 1);

    // extract key points using sift algorithm
    cout<<"extracting feature points ......"<<endl;

    cv::SiftFeatureDetector detector;
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desp1, desp2;
    detector.detect(img1, kp1);
    detector.detect(img2, kp2);
    cv::SiftDescriptorExtractor extractor;
    extractor.compute(img1, kp1, desp1);
    extractor.compute(img2, kp2, desp2);

    cout<<"we get "<<kp1.size()<<" and "<<kp2.size()<<" key points! "<<endl;
    
    // match key points using knnMatch and RANSAC
    cout<<"matching feature points ......"<<endl;
    cv::FlannBasedMatcher matcher;
    vector<PairMatches> matchesInfo_2nn;
    matcher.knnMatch(desp1, desp2, matchesInfo_2nn, 2);
    PairMatches goodmatches;
    for (int i = 0; i < matchesInfo_2nn.size(); ++i)
    {
        if (matchesInfo_2nn[i][0].distance < 0.6*matchesInfo_2nn[i][1].distance)
        {
            goodmatches.push_back(matchesInfo_2nn[i][0]);
        }
    }
    vector<cv::Point2f> temp1, temp2;
    for (int i = 0; i < goodmatches.size(); ++i)
    {
        temp1.push_back(kp1[goodmatches[i].queryIdx].pt);
        temp2.push_back(kp2[goodmatches[i].trainIdx].pt);
    }
    cv::Mat ransac_mask;
    cv::Mat fundamental = cv::findFundamentalMat(temp1, temp2, ransac_mask, CV_FM_RANSAC);
    int inlier_count = 0;
    for (int i = 0; i < ransac_mask.rows; ++i)
    {
        if (ransac_mask.at<uchar>(i) == 1)
        {
            points1.push_back(temp1[i]);
            points2.push_back(temp2[i]);
            inlier_count++;
        }
    }
    cout<<"we get "<<inlier_count<<" match points! "<<endl;

    if (inlier_count<32)
    {
        cout<<"too few match points. return false!"<<endl;
        return false;
    }

    return true;
}
