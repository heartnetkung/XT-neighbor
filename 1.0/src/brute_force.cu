#include <unordered_set>
#include "codec.cu"

struct Triplet {
    int x, y, z;
    Triplet(int x2, int y2, int z2) {
        x = x2;
        y = y2;
        z = z2;
    }
    bool operator==(const Triplet& t) const {
        return (x == t.x) && (y == t.y) && (z == t.z);
    }
};

struct TripletHasher {
    size_t operator()(const Triplet& i) const {
        return i.x + i.y + i.z;
    }
};

// source: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

int levenshtein(char *s1, char *s2) {
    int s1len, s2len, x, y, lastdiag, olddiag;
    s1len = strlen(s1);
    s2len = strlen(s2);
    int column[s1len + 1];
    for (y = 1; y <= s1len; y++)
        column[y] = y;
    for (x = 1; x <= s2len; x++) {
        column[0] = x;
        for (y = 1, lastdiag = x - 1; y <= s1len; y++) {
            olddiag = column[y];
            column[y] = MIN3(column[y] + 1, column[y - 1] + 1, lastdiag + (s1[y - 1] == s2[x - 1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return column[s1len];
}

std::unordered_set<Triplet, TripletHasher> pairwise_distance(Int3* inputs, int len, int threshold) {
    char* strings[len];
    for (int i = 0; i < len; i++)
        strings[i] = str_decode(inputs[i]);

    std::unordered_set<Triplet, TripletHasher> ans;
    for (int i = 0; i < len; i++)
        for (int j = i + 1; j < len; j++) {
            int distance = levenshtein(strings[i], strings[j]);
            if (distance <= threshold) {
                Triplet newOutput(i, j, distance);
                ans.insert(newOutput);
            }
        }

    return ans;
}

int check_intput(std::unordered_set<Triplet, TripletHasher> answer, SymspellOutput output) {
    if (output.len != answer.size())
        return 0;

    Int2 current;
    std::unordered_set<Triplet, TripletHasher> inputs;
    for (int i = 0; i < output.len; i++) {
        current = output.indexPairs[i];
        Triplet value(current.x, current.y, output.pairwiseDistances[i]);
        if (!answer.count(value))
            return 0;
        inputs.insert(value);
    }

    if (answer.size() != inputs.size())
        return 0;

    return 1;
}
