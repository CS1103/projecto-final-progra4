#include <iostream>
#include <random>
#include <limits>
#include <chrono>
#include "neural_network.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"

using namespace utec::neural_network;
using namespace std::chrono;

void clearInputBuffer() {
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::pair<Tensor<float,2>, Tensor<float,2>> generate_data(size_t samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 99);

    Tensor<float,2> X(samples, 2);
    Tensor<float,2> Y(samples, 1);

    for (size_t i = 0; i < samples; ++i) {
        int a = dis(gen);
        int b = dis(gen);
        X.at(i, 0) = static_cast<float>(a);
        X.at(i, 1) = static_cast<float>(b);
        Y.at(i, 0) = static_cast<float>(a + b);
    }

    return {X, Y};
}

int MainMenu() {
    int choice;
    do {
        std::cout << "\n=== MENU PRINCIPAL ===\n";
        std::cout << "1. Entrenar y probar con valores por defecto\n";
        std::cout << "2. Entrenar y probar con parametros personalizados\n";
        std::cout << "3. Salir\n";
        std::cout << "Seleccione una opcion: ";
        std::cin >> choice;

        if (std::cin.fail() || choice < 1 || choice > 3) {
            std::cin.clear();
            clearInputBuffer();
            std::cout << "Opcion no valida. Intente nuevamente.\n";
        } else {
            break;
        }
    } while (true);

    clearInputBuffer();
    return choice;
}

void trainWithDefaultParams() {
    std::cout << "\n=== ENTRENAMIENTO CON PARAMETROS POR DEFECTO ===\n";
    std::cout << "Parametros:\n";
    std::cout << "- Capas: 3 (64, 32 y 16 neuronas)\n";
    std::cout << "- Funciones de activacion: ReLU\n";
    std::cout << "- Epocas: 15\n";
    std::cout << "- Tasa de aprendizaje: 0.001\n";
    std::cout << "- Tamano de lote: 32\n";

    auto [X_train, Y_train] = generate_data(1000);
    auto [X_test, Y_test] = generate_data(1000);

    //Normalización
    X_train *= (1.0f / 99.0f);
    Y_train *= (1.0f / 198.0f);
    X_test *= (1.0f / 99.0f);
    Y_test *= (1.0f / 198.0f);

    NeuralNetwork<float> nn;
    nn.add_layer(std::make_unique<Dense<float>>(2, 64));
    nn.add_layer(std::make_unique<ReLU<float>>());
    nn.add_layer(std::make_unique<Dense<float>>(64, 32));
    nn.add_layer(std::make_unique<ReLU<float>>());
    nn.add_layer(std::make_unique<Dense<float>>(32, 16));
    nn.add_layer(std::make_unique<ReLU<float>>());
    nn.add_layer(std::make_unique<Dense<float>>(16, 1));

    nn.set_optimizer(std::make_unique<Adam<float>>(0.001));

    auto start_time = high_resolution_clock::now();

    std::cout << "\n=== PROCESO DE ENTRENAMIENTO ===\n";
    std::cout << "Epoca\tPerdida\n";
    std::cout << "-----------------\n";
    nn.train(X_train, Y_train, 15, 32);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    std::cout << "\n=== TIEMPO DE ENTRENAMIENTO ===\n";
    std::cout << "Tiempo total: " << duration.count() << " milisegundos\n";
    std::cout << "Tiempo por epoca: " << duration.count()/15.0 << " ms/epoca\n";

    std::cout << "\n=== PRUEBAS AUTOMATICAS ===\n";
    std::vector<std::pair<int, int>> test_cases = {
        {1,1}, {4,23}, {7,8}, {10,20}, {50,50}, {99,99},
        {12,45}, {78,21}, {5,95}, {33,66}, {9,89}, {45,55}
    };
    int correct = 0;
    for (auto [a, b] : test_cases) {
        Tensor<float,2> input(1, 2);
        input.at(0, 0) = a / 99.0f;
        input.at(0, 1) = b / 99.0f;

        auto output = nn.forward(input);
        float predicted = output.at(0,0) * 198.0f;

        std::cout << "\n=== SUMA ===\n";
        std::cout << a << " + " << b << " = ?\n";
        std::cout << "Prediccion de la red: " << predicted << "\n";
        std::cout << "Resultado real: " << (a + b) << "\n";
        std::cout << "Error: " << std::abs(predicted - (a + b)) << "\n";
        std::cout << "Error porcentual: "
                  << (std::abs(predicted - (a + b)) / (a + b)) * 100.0f
                  << "%\n";
        if (std::abs(predicted - (a + b)) <= 0.3) {
            correct++;
        }
    }
    std::cout << "\n=== RESUMEN DE PRECISION ===\n";
    float accuracy = (static_cast<float>(correct) / test_cases.size()) * 100.0f;
    std::cout << "Precision total: " << accuracy << "%\n";
}

void getCustomParam(size_t& epochs, size_t& batch_size, float& learning_rate,
                   size_t& layer1_size, size_t& layer2_size, size_t& layer3_size) {
    std::cout << "\n=== CONFIGURACION PERSONALIZADA ===\n";
    std::cout << "Ingrese los parametros (presione Enter para usar valores por defecto):\n";

    std::cout << "Numero de epocas (default 15): ";
    std::string input;
    std::getline(std::cin, input);
    epochs = input.empty() ? 15 : std::stoul(input);

    std::cout << "Tamano de lote (default 32): ";
    std::getline(std::cin, input);
    batch_size = input.empty() ? 32 : std::stoul(input);

    std::cout << "Tasa de aprendizaje (default 0.001): ";
    std::getline(std::cin, input);
    learning_rate = input.empty() ? 0.001f : std::stof(input);

    std::cout << "Neuronas en primera capa oculta (default 64): ";
    std::getline(std::cin, input);
    layer1_size = input.empty() ? 64 : std::stoul(input);

    std::cout << "Neuronas en segunda capa oculta (default 32): ";
    std::getline(std::cin, input);
    layer2_size = input.empty() ? 32 : std::stoul(input);

    std::cout << "Neuronas en tercera capa oculta (default 16): ";
    std::getline(std::cin, input);
    layer3_size = input.empty() ? 16 : std::stoul(input);
}

void trainWithCustomParams() {
    size_t epochs, batch_size, layer1_size, layer2_size, layer3_size;
    float learning_rate;

    getCustomParam(epochs, batch_size, learning_rate, layer1_size, layer2_size, layer3_size);

    std::cout << "\n=== RESUMEN DE CONFIGURACION ===\n";
    std::cout << "Epocas: " << epochs << "\n";
    std::cout << "Tamano de lote: " << batch_size << "\n";
    std::cout << "Tasa de aprendizaje: " << learning_rate << "\n";
    std::cout << "Neuronas capa 1: " << layer1_size << "\n";
    std::cout << "Neuronas capa 2: " << layer2_size << "\n";
    std::cout << "Neuronas capa 3: " << layer3_size << "\n";

    auto [X_train, Y_train] = generate_data(1000);
    auto [X_test, Y_test] = generate_data(1000);

    // Normalización para 0-99
    X_train *= (1.0f / 99.0f);
    Y_train *= (1.0f / 198.0f);
    X_test *= (1.0f / 99.0f);
    Y_test *= (1.0f / 198.0f);

    NeuralNetwork<float> nn;
    nn.add_layer(std::make_unique<Dense<float>>(2, layer1_size));
    nn.add_layer(std::make_unique<ReLU<float>>());
    nn.add_layer(std::make_unique<Dense<float>>(layer1_size, layer2_size));
    nn.add_layer(std::make_unique<ReLU<float>>());
    nn.add_layer(std::make_unique<Dense<float>>(layer2_size, layer3_size));
    nn.add_layer(std::make_unique<ReLU<float>>());
    nn.add_layer(std::make_unique<Dense<float>>(layer3_size, 1));

    nn.set_optimizer(std::make_unique<Adam<float>>(learning_rate));

    auto start_time = high_resolution_clock::now();

    std::cout << "\n=== PROCESO DE ENTRENAMIENTO ===\n";
    std::cout << "Epoca\tPerdida\n";
    std::cout << "-----------------\n";
    nn.train(X_train, Y_train, epochs, batch_size);

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);

    std::cout << "\n=== TIEMPO DE ENTRENAMIENTO ===\n";
    std::cout << "Tiempo total: " << duration.count() << " milisegundos\n";
    std::cout << "Tiempo por epoca: " << duration.count()/static_cast<float>(epochs) << " ms/epoca\n";

    std::cout << "\n=== PRUEBAS AUTOMATICAS ===\n";
    std::vector<std::pair<int, int>> test_cases = {
        {1,1}, {4,23}, {7,8}, {10,20}, {50,50}, {99,99},
        {12,45}, {78,21}, {5,95}, {33,66}, {9,89}, {45,55}
    };
    int correct = 0;
    for (auto [a, b] : test_cases) {
        Tensor<float,2> input(1, 2);
        input.at(0, 0) = a / 99.0f;
        input.at(0, 1) = b / 99.0f;

        auto output = nn.forward(input);
        float predicted = output.at(0,0) * 198.0f;

        std::cout << "\n=== SUMA ===\n";
        std::cout << a << " + " << b << " = ?\n";
        std::cout << "Prediccion de la red: " << predicted << "\n";
        std::cout << "Resultado real: " << (a + b) << "\n";
        std::cout << "Error: " << std::abs(predicted - (a + b)) << "\n";
        std::cout << "Error porcentual: "
                  << (std::abs(predicted - (a + b)) / (a + b)) * 100.0f
                  << "%\n";
        if (std::abs(predicted - (a + b)) <= 0.3) {
            correct++;
        }
    }
    std::cout << "\n=== RESUMEN DE PRECISION ===\n";
    float accuracy = (static_cast<float>(correct) / test_cases.size()) * 100.0f;
    std::cout << "Precision total: " << accuracy << "%\n";

    std::cout << "\n=== MODO INTERACTIVO ===\n";
    std::cout << "Ingrese dos numeros entre 0 y 99 para que la red los sume (o -1 para salir)\n";

    while (true) {
        int a, b;
        std::cout << "\nPrimer numero (0-99): ";
        std::cin >> a;

        if (a == -1) break;
        if (a < 0 || a > 99) {
            std::cout << "Numero fuera de rango. Intente nuevamente.\n";
            clearInputBuffer();
            continue;
        }

        std::cout << "Segundo numero (0-99): ";
        std::cin >> b;

        if (b == -1) break;
        if (b < 0 || b > 99) {
            std::cout << "Numero fuera de rango. Intente nuevamente.\n";
            clearInputBuffer();
            continue;
        }

        clearInputBuffer();

        Tensor<float,2> input(1, 2);
        input.at(0, 0) = a / 99.0f;
        input.at(0, 1) = b / 99.0f;

        auto output = nn.forward(input);
        float predicted = output.at(0,0) * 198.0f;

        std::cout << "\n=== SUMA ===\n";
        std::cout << a << " + " << b << " = ?\n";
        std::cout << "Prediccion de la red: " << predicted << "\n";
        std::cout << "Resultado real: " << (a + b) << "\n";
        std::cout << "Error: " << std::abs(predicted - (a + b)) << "\n";
        std::cout << "Error porcentual: "
                  << (std::abs(predicted - (a + b)) / (a + b)) * 100.0f
                  << "%\n";
    }
}

int main() {
    std::cout << "RED NEURONAL PARA SUMAR NUMEROS DE 2 DIGITOS (0-99)\n";

    while (true) {
        int choice = MainMenu();

        switch (choice) {
            case 1:
                trainWithDefaultParams();
                break;
            case 2:
                trainWithCustomParams();
                break;
            case 3:
                std::cout << "Saliendo del programa...\n";
                return 0;
        }
    }

    return 0;
}