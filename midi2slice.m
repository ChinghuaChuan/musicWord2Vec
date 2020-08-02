function onset_mat = midi2slice(midi_file, beat, pc)

% midi2slice: convering a midi file to a sequence of music slices
% This function requires MIDIToolbox, https://github.com/miditoolbox/1.1.
% Example: onset_mat = midi2slice('wtc1p01.mid');
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



    if nargin < 2
       beat = 1;
       pc = 1;
    elseif nargin < 3
       pc = 1;
    end


    midimat = readmidi(midi_file);
    % clean up the duration 
    midimat(:, 1) = round(midimat(:, 1)*1000)/1000;
    midimat(:, 2) = round(midimat(:, 2)*1000)/1000;
    
    num_slices = floor(max(midimat(:, 1)+midimat(:, 2))/beat)+1;
    
    % onset_mat: num_slices x num_pitches or num_pitch_classes,
    % 88 keys: from A0 to C8
    highest_pitch = 108; %C8
    lowest_pitch = 21; %A0
    if pc == 1 % 12 pitch classes
        onset_mat = zeros(num_slices, 12);
    else % 88 keys
        onset_mat = zeros(num_slices, highest_pitch - lowest_pitch + 1);
    end
    
    % For each pitch, [begin_slice end_slice] 
    slice_mat = [floor(midimat(:, 1)/beat)+1 floor((midimat(:, 1)+midimat(:, 2))/beat)+1];
    
    % for pitch that starts and ends in the same slice
    index = find((slice_mat(:, 1) - slice_mat(:, 2)) == 0);
    if ~isempty(index)
        value = ones(length(index), 1);
        if pc == 1
            rowNos = slice_mat(index, 1);
            colNos = mod(midimat(index, 4), 12) + 1;
            onset_mat(sub2ind(size(onset_mat), rowNos, colNos)) = value;
        else
            index2 = find((midimat(index, 4)>= lowest_pitch) & (midimat(index, 4)<=highest_pitch));
            
            if ~isempty(index2)
                rowNos = slice_mat(index(index2), 1);
                colNos = (midimat(index(index2), 4) - lowest_pitch + 1);
                onset_mat(sub2ind(size(onset_mat), rowNos, colNos)) = value(index2);
            end
        end
    end
    
    % for pitch that is played in several slices
    index = find(abs(slice_mat(:, 1) - slice_mat(:, 2))>0);
    
    if ~isempty(index)
        for i = 1:length(index)
           begin_slice = slice_mat(index(i), 1);
           end_slice = slice_mat(index(i), 2);
           
           value = 1;

           if pc == 1
               onset_mat(begin_slice, mod(midimat(index(i), 4), 12) + 1) = value;
           else
               if (midimat(index(i), 4) >= lowest_pitch) && (midimat(index(i), 4) <= highest_pitch) 
                   onset_mat(begin_slice, midimat(index(i), 4) - lowest_pitch + 1) = value;
               end    
           end
           
           value = 1;

           if pc == 1
               onset_mat(end_slice, mod(midimat(index(i), 4), 12) + 1) = value;
           else
               if (midimat(index(i), 4) >= lowest_pitch) && (midimat(index(i), 4) <= highest_pitch) 
                   onset_mat(end_slice, midimat(index(i), 4) - lowest_pitch + 1) = value;
               end    
           end
           
           % in-between slices
           if end_slice - begin_slice > 0
              slice_index = begin_slice+1:end_slice-1;
              value = 1;
              if pc == 1
                  onset_mat(slice_index, mod(midimat(index(i), 4), 12) + 1) = value;
              else
                  if (midimat(index(i), 4) >= lowest_pitch) && (midimat(index(i), 4) <= highest_pitch) 
                      onset_mat(slice_index, midimat(index(i), 4) - lowest_pitch + 1) = value;
                  end 
              end
           end
        end
    end
    