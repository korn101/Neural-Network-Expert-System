function CLAMPED_RANDOM = clampedRandom()

    noHiddenNeuronsInNetwork = GetGlobalHidden();
    CLAMPED_RANDOM = (2*(rand./sqrt(noHiddenNeuronsInNetwork)) - (1./sqrt(noHiddenNeuronsInNetwork)));

end