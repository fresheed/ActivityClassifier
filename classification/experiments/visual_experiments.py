from classification.preparation import get_classified_chunks, split_items_set
from classification.experiments.experiments import Experiment, display_accuracy, display_chunks_stats, to_timedelta
from classification.features import mlp, fft
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
from scipy.signal import argrelextrema


class FFTVisualExperiment(Experiment):
    
    def run(self, log_dir, classes):
        classified_chunks=get_classified_chunks(log_dir, classes, 
                                                to_timedelta(3))

        # train_set, test_set=split_items_set(classified_chunks)
        # display_chunks_stats(classes, train_set, test_set)

        # extractor=fft.FFTCoeffsExtractor()
        
        # classifier=mlp.MLPClassifier(extractor)

        # confmat=self.explore_classifier(classifier, train_set, test_set)
        # display_accuracy(confmat)
        even_frame=classified_chunks[0][0]
        
        freqs=np.fft.fftfreq(len(even_frame.values))
        spectrum=np.fft.fftn(even_frame.values)
        magnitudes=abs(spectrum[:, 0])
        up_to=len(freqs)//2
        plot_freqs=freqs[1:up_to]
        plot_mags=magnitudes[1:up_to]
        def is_peak(ind):
            cur=magnitudes[ind]
            left=magnitudes[ind-1]
            right=magnitudes[ind+1]
            diff_left=cur-left
            diff_right=cur-right
            threshold=cur/20
            return ((diff_left>threshold and diff_right>threshold)
                    or (diff_left<-threshold and diff_right<-threshold))
        extrems_at=[ind-1 for ind in range(1, len(freqs)-1)
                    if is_peak(ind)]

        plot_extrems=[plot_mags[ind] if ind in extrems_at else 0
                      for ind in range(0, len(plot_mags)-1)]
        plt.plot(plot_freqs, plot_mags)
        plt.plot(plot_freqs[:-1], plot_extrems)
        plt.show()


if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_1"]
    FFTVisualExperiment().run(log_dir, classes)
