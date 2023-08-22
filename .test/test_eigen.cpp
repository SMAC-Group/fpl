#include <iostream>
#include <random>
#include <iterator>
#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

int main(){
    VectorXi v(5);
    VectorXd x = VectorXd::Random(5);
    v << 2, 3, 2, 1, 0;
    std::cout << x << std::endl;
    std::cout << v.unaryExpr(x) << std::endl;
}

// int main(){
//     //Eigen::VectorXi ivec = Eigen::VectorXi::LinSpaced(5,0,4);
//     std::vector<int> v(5);
//     std::iota(v.begin(), v.end(), 0);
//     std::random_device rd;
//     std::mt19937 g(rd());
//     std::shuffle(v.begin(), v.end(), g);
//     //Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(5);
//     //perm.setIdentity();
//     //std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
//     //std::cout << v << std::endl;
//     std::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
//     std::cout << "\n";
// }
