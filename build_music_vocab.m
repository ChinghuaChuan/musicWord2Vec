function build_music_vocab(folder, beat, pc)
% build_music_vocab: Building the word2vec "vocabulary" using music slices
% from all the midi files in the folder
% This function calls midi2slice.m and requires Matlab Communications Toolbox.
% Example: build_music_vocab('.\', 1, 1);
% beat = 1: music slice is generated at every beat, examining the presence of each pitch in the beat.
% pc = 1: music slice is based on 12 pitch classes, pc = 0: pitchs from A0
% to C8
%
% More information on music slices and music word2vec, refer to 
% Chuan C.H., Agres, K., and Herremans D., "From Context to Concept: Exploring Semantic Relationships in Music with Word2Vec,"
% Neural Computing and Applications, special issue on Deep Learning for Music and Audio, Springer, 32(4), 2020, pp. 1023-1036,
% DOI: https://doi.org/10.1007/s0052.

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.


files = dir(fullfile(folder, '*.mid'));
fid = fopen('music_slice.txt', 'a');
fid_log = fopen('music_slice_file_list.txt', 'a');

if nargin < 2
   beat = 1;
   pc = 1;
elseif nargin < 3
   pc = 1;
end

if pc == 1
    vocab_mat = zeros(pow2(13), 1); 
else
    vocab_mat = zeros(1, 10); 
end


for i = 1:length(files)
    midifile = files(i).name
    
    onset_mat = midi2slice(midifile, beat, pc);
    if ~isempty(onset_mat)
        % 12 pitch classes
        if pc == 1
             dec_mat = bi2de(onset_mat);
             output = [];
             for j = 1:length(dec_mat)
                  vocab_mat(dec_mat(j)+1) = vocab_mat(dec_mat(j)+1) + 1;
                  output = [output dec_mat(j)];
             end
        else % 88 keys: A0 to C8
            dec_mat1 = bi2de(onset_mat(:, 1:3)); %A0, .., B0
            dec_mat2 = bi2de(onset_mat(:, 4:15)); %C1...
            dec_mat3 = bi2de(onset_mat(:, 16:27)); %C2...
            dec_mat4 = bi2de(onset_mat(:, 28:39)); %C3...
            dec_mat5 = bi2de(onset_mat(:, 40:51)); %C4...
            dec_mat6 = bi2de(onset_mat(:, 52:63)); %C5...
            dec_mat7 = bi2de(onset_mat(:, 64:75)); %C6...
            dec_mat8 = bi2de(onset_mat(:, 76:87)); %C7...
            dec_mat9 = bi2de(onset_mat(:, 88)); %C8

            dec_mat = [dec_mat1 dec_mat2 dec_mat3 dec_mat4 dec_mat5 dec_mat6 dec_mat7 dec_mat8 dec_mat9];
            output = '';
            for j = 1:size(dec_mat, 1)
                [rows index] = intersect(vocab_mat(:, 1:9), dec_mat(j, 1:9), 'rows');
                if ~isempty(index)
                    %disp 'found existing word'
                    vocab_mat(index, 10) = vocab_mat(index, 10)+1;
                else
                    %disp 'add new word'
                    vocab_mat = [vocab_mat; dec_mat(j, :) 1];
                end
                if j == 1
                    output = strcat(int2str(dec_mat(j, 1)), '-', int2str(dec_mat(j, 2)), '-',...
                    int2str(dec_mat(j, 3)), '-', int2str(dec_mat(j, 4)), '-', int2str(dec_mat(j, 5)), '-',...
                    int2str(dec_mat(j, 6)), '-', int2str(dec_mat(j, 7)), '-', int2str(dec_mat(j, 8)), '-',...
                    int2str(dec_mat(j, 9)));
                else
                    output = strcat(output, ', ', int2str(dec_mat(j, 1)), '-', int2str(dec_mat(j, 2)), '-',...
                    int2str(dec_mat(j, 3)), '-', int2str(dec_mat(j, 4)), '-', int2str(dec_mat(j, 5)), '-',...
                    int2str(dec_mat(j, 6)), '-', int2str(dec_mat(j, 7)), '-', int2str(dec_mat(j, 8)), '-',...
                    int2str(dec_mat(j, 9)));
                end
            end
        end
        fprintf(fid, '%d ', output);
        fprintf(fid, '\n');
        fprintf(fid_log, '%s ', midifile);
        fprintf(fid_log, '\n');
    end
    
    if mod(i, 5000) == 0
        dlmwrite('vocab_slice_occurrence.txt', vocab_mat);
    end
end

dlmwrite('vocab_slice_occurrence.txt', vocab_mat);
fclose(fid);
fclose(fid_log);




