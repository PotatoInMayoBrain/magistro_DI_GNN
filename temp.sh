awk '{
    split($0, distances, " ");
    for (i = 1; i <= length(distances); i++) {
        if (distances[i] == -1) {
            if (i == 1 || prev != -1) {
                printf "\n";
            }
        } else {
            printf "%s ", distances[i];
        }
        prev = distances[i];
    }
}
END {
    printf "\n";
}' temp2
