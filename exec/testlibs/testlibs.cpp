// This file tests if third party libraries are working correctly

#include <iostream>
#include <fstream>
#include <mlpack/core.hpp>
#include <armadillo>

#include <json/json.h>

using namespace std;
using namespace arma;
using namespace mlpack;

void test_jsoncpp(){
    cout << "start test_jsoncpp" << endl;

    ifstream ifs("data/jsoncpp_test.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj); // reader can also read strings
    cout << " Book: " << obj["book"].asString() << endl;
    cout << " Year: " << obj["year"].asUInt() << endl;
    const Json::Value& characters = obj["characters"]; // array of characters
    for (int i = 0; i < characters.size(); i++){
        cout << "    name: " << characters[i]["name"].asString();
        cout << " chapter: " << characters[i]["chapter"].asUInt();
        cout << endl;
    }

    cout << "end test_jsoncpp" << endl;
}


void test_armadillo(){
    cout << "start test_armadillo" << endl;

    mat A(4, 5, fill::randu);
    mat B(4, 5, fill::randu);
    cout << A*B.t() << endl;

    cout << "end test_armadillo" << endl;
}


void test_mlpack(){
    cout << "start test_mlpack" << endl;

    arma::mat data;
    data::Load("data/data.csv", data, true);
    arma::mat cov = data * trans(data) / data.n_cols;
    data::Save("data/cov.csv", cov, true);

    cout << "end test_mlpack" << endl;
}


int main(int argc, char *argv[]){

    cout << "Testing third-party libraries" << endl;

    test_jsoncpp();

    test_armadillo();
//
    test_mlpack();

    return 0;


}


