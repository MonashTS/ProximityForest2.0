#include <exception>
#include <regex>

#include <tempo/utils/readingtools.hpp>
#include <tempo/dataset/dts.hpp>
#include <tempo/reader/dts.reader.hpp>
#include <tempo/transform/tseries.univariate.hpp>
#include <tempo/classifier/TSChief/forest.hpp>

#include <nlohmann/json.hpp>
#include "cmdline.hpp"
#include "tempo/classifier/TSChief/pfsplitters.hpp"

#include "tempo/classifier/ProximityForest2/pf2.hpp"

using namespace std;
using namespace tempo;

namespace fs = std::filesystem;

[[noreturn]] void do_exit(int code, std::optional<std::string> msg = {}) {
    if (msg) { std::cerr << msg.value() << std::endl; }
    exit(code);
}

cmdopt getcmdopt(int argc, char **argv) {
    cmdopt opt;
    variant<string, cmdopt> mb_opt = parse_cmd(argc, argv);
    switch (mb_opt.index()) {
        case 0: {
            cerr << "Error: " << std::get<0>(mb_opt) << std::endl;
            exit(1);
        }
        case 1: {
            opt = std::get<1>(mb_opt);
        }
    }
    return opt;
}


int main(int argc, char **argv) {

    // --- --- --- Type / namespace

    namespace tsc = tempo::classifier::TSChief;
    namespace tsc_nn1 = tempo::classifier::TSChief::snode::nn1splitter;

    // --- --- --- Randomness
    std::random_device rd;
    size_t state_seed = rd();
    size_t tiebreak_seed = rd();

    // --- --- --- Prepare JSon record for output
    nlohmann::json jv;

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Read args
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    cmdopt opt = getcmdopt(argc, argv);

    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Read dataset
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    DTS train_dataset;
    DTS test_dataset;
    {
        using namespace tempo::reader::dataset;
        Result read_dataset_result;
        read_dataset_result = load(opt.input);

        if (read_dataset_result.index() == 0) { do_exit(1, std::get<0>(read_dataset_result)); }
        TrainTest traintest = std::get<1>(std::move(read_dataset_result));
        train_dataset = traintest.train_dataset;
        test_dataset = traintest.test_dataset;

        nlohmann::json dataset;
        dataset["train"] = train_dataset.header().to_json();
        dataset["test"] = test_dataset.header().to_json();
        dataset["load_time_ns"] = traintest.load_time.count();
        dataset["load_time_str"] = utils::as_string(traintest.load_time);
        jv["dataset"] = dataset;

        // --- --- --- Sanity check
        std::vector<std::string> errors = sanity_check(traintest);

        if (!errors.empty()) {
            jv["status"] = "error";
            jv["status_message"] = utils::cat(errors, "; ");
            cout << to_string(jv) << endl;
            if (opt.output) {
                auto out = ofstream(opt.output.value());
                out << jv << endl;
            }
            exit(1);
        }
    } // End of dataset loading

    DatasetHeader const &train_header = train_dataset.header();
    DatasetHeader const &test_header = test_dataset.header();

    // --- --- ---
    // --- --- --- STATE
    // --- --- ---

    std::cout << "State seed = " << state_seed << std::endl;
    tsc::TreeState tstate(state_seed, 0);

    // --- --- ---
    // --- --- --- Train the tree
    // --- --- ---
    tempo::classifier::ProximityForest2 classifier = tempo::classifier::ProximityForest2(
            train_dataset,
            train_header,
            opt.nb_candidates,
            opt.nb_trees,
            tstate
    );
    classifier.train(opt.nb_threads);

    // --- --- --- TEST

    classifier::ResultN result = classifier.predict(test_dataset, opt.nb_threads);

    PRNG prng(tiebreak_seed);
    size_t nb_correct = result.nb_correct_01loss(test_header, IndexSet(test_header.size()), prng);
    double accuracy = (double) nb_correct / (double) test_header.size();

    if (opt.prob_output) {
        arma::field<std::string> header(test_header.nb_classes());
        for (size_t i = 0; i < test_header.nb_classes(); ++i) { header(i) = test_header.decode(i); }
        result.probabilities.save(arma::csv_name(opt.prob_output.value(), header));
    }


    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    // Generate output and exit
    // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---


    jv["status"] = "success";

    { // Classifier information
        auto prepare_data_elapsed = classifier.prepare_train_data_time + classifier.prepare_test_data_time;

        nlohmann::json j;
        j["train_time_ns"] = classifier.train_time.count();
        j["train_time_human"] = utils::as_string(classifier.train_time);
        j["test_time_ns"] = classifier.test_time.count();
        j["test_time_human"] = utils::as_string(classifier.test_time);
        j["prepare_data_ns"] = prepare_data_elapsed.count();
        j["prepare_data_human"] = utils::as_string(prepare_data_elapsed);
        //
        jv["classifier"] = opt.pfconfig;
        jv["classifier_info"] = j;
    }

    { // 01 loss results
        nlohmann::json j;
        j["nb_corrects"] = (int) nb_correct;
        j["accuracy"] = accuracy;
        jv["01loss"] = j;
    }

    cout << jv.dump(2) << endl;

    if (opt.output) {
        auto out = ofstream(opt.output.value());
        out << jv.dump(2) << endl;
    }

    return 0;

}